"""Fit the ensemble's blend weights by walk-forward backtesting the four
component models (Monte Carlo, Dixon-Coles, XGBoost, squad-strength proxy)
against major-tournament matches since 2010."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TOURNAMENT_AVG_XG, DIXON_COLES_RHO, MAX_GOALS, ELO_BASE
from .team_model import predict_symmetric
from .poisson_model import score_probability_matrix, extract_probabilities

DEFAULT_WEIGHTS = (0.35, 0.30, 0.20, 0.15)


def _simulate_component_predictions(row, final_elo, long, model, knockout_regression=0.95):
    """
    For a single historical match, compute what each of the 4 component
    models would have predicted (3-way: H/D/A probabilities).

    Components:
      1. Monte Carlo (Poisson with xG-derived lambdas)
      2. Dixon-Coles (same lambdas)
      3. XGBoost
      4. Squad-strength proxy (Elo-based sigmoid)

    Returns (p_sim, p_pois, p_xgb, p_squad) each as (h, d, a) tuple,
    or None if we can't compute for this match.
    """
    home = row["home_team"]
    away = row["away_team"]
    date_str = str(row["date"].date())

    # ── XGBoost prediction ──
    try:
        p_xgb_h, p_xgb_d, p_xgb_a = predict_symmetric(
            model, long, final_elo, home, away, date_str,
            neutral=int(row.get("neutral", 1)), weight=4)
    except Exception:
        return None

    # ── Elo-based proxy for "squad strength" ──
    h_elo = final_elo.get(home, ELO_BASE)
    a_elo = final_elo.get(away, ELO_BASE)
    elo_diff = h_elo - a_elo
    p_sq_raw_h = 1.0 / (1.0 + np.exp(-elo_diff / 200.0))
    p_sq_raw_a = 1.0 - p_sq_raw_h
    p_sq_d = 0.22
    p_sq_h = p_sq_raw_h * (1.0 - p_sq_d)
    p_sq_a = p_sq_raw_a * (1.0 - p_sq_d)

    # ── xG-derived lambdas for Monte Carlo / Dixon-Coles ──
    # Use Elo-derived attack/defense strengths as proxy (no player data for
    # historical matches). This is the best we can do with results.csv.
    h_att_proxy = max(0.5, 1.0 + elo_diff / 800.0)  # >1 if strong
    a_att_proxy = max(0.5, 1.0 - elo_diff / 800.0)
    h_def_proxy = max(0.3, 1.0 - elo_diff / 1200.0)  # lower = better defense
    a_def_proxy = max(0.3, 1.0 + elo_diff / 1200.0)

    lambda_h = TOURNAMENT_AVG_XG * h_att_proxy * a_def_proxy * knockout_regression
    lambda_a = TOURNAMENT_AVG_XG * a_att_proxy * h_def_proxy * knockout_regression
    lambda_h = np.clip(lambda_h, 0.3, 4.0)
    lambda_a = np.clip(lambda_a, 0.3, 4.0)

    # Monte Carlo
    goals_h = np.random.poisson(lambda_h, 2000)
    goals_a = np.random.poisson(lambda_a, 2000)
    p_sim_h = np.mean(goals_h > goals_a)
    p_sim_d = np.mean(goals_h == goals_a)
    p_sim_a = np.mean(goals_h < goals_a)

    # Dixon-Coles
    matrix = score_probability_matrix(lambda_h, lambda_a, DIXON_COLES_RHO, MAX_GOALS)
    p_pois_h, p_pois_d, p_pois_a = extract_probabilities(matrix)

    return (
        (p_sim_h, p_sim_d, p_sim_a),
        (p_pois_h, p_pois_d, p_pois_a),
        (p_xgb_h, p_xgb_d, p_xgb_a),
        (p_sq_h, p_sq_d, p_sq_a),
    )


def _compute_calibration_bins(predicted_probs, actual_outcomes, n_bins=5):
    """Compute calibration data for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    cal_data = []
    for i in range(n_bins):
        mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
        if mask.sum() > 2:
            cal_data.append({
                "bin_center": (bins[i] + bins[i + 1]) / 2,
                "predicted_mean": float(predicted_probs[mask].mean()),
                "actual_mean": float(actual_outcomes[mask].mean()),
                "count": int(mask.sum()),
            })
    return cal_data


def fit_ensemble_weights(dataset, final_elo, long, model, knockout_regression=0.95):
    """
    Walk-forward backtest on major-tournament matches since 2010.
    Fit blend weights to minimize log-loss via scipy.optimize.

    Returns:
        weights: (w_sim, w_pois, w_xgb, w_squad)
        backtest_results: dict with metrics and per-match predictions
    """
    backtest = dataset[
        (dataset["date"] >= pd.Timestamp("2010-01-01")) &
        (dataset["date"] < pd.Timestamp("2026-01-01"))
    ].copy()

    backtest = backtest[backtest["tournament"].apply(
        lambda t: any(k in str(t).lower() for k in
                      ["fifa world cup", "uefa euro", "copa america",
                       "africa cup", "confederations"])
    )].copy()

    if len(backtest) < 30:
        print("  ⚠️  WARNING: <30 backtest matches, using hand-picked defaults")
        return DEFAULT_WEIGHTS, None

    predictions = []  # list of (actual_label, (sim, pois, xgb, squad))
    for _, row in backtest.iterrows():
        result = _simulate_component_predictions(row, final_elo, long, model, knockout_regression)
        if result is None:
            continue
        actual = int(row["label"])  # 0=home_win, 1=draw, 2=away_win
        predictions.append((actual, result))

    n_matches = len(predictions)
    if n_matches < 20:
        print(f"  ⚠️  WARNING: only {n_matches} backtest predictions, using defaults")
        return DEFAULT_WEIGHTS, None

    actuals = np.array([p[0] for p in predictions])
    sim_probs = np.array([p[1][0] for p in predictions])   # (N, 3)
    pois_probs = np.array([p[1][1] for p in predictions])
    xgb_probs = np.array([p[1][2] for p in predictions])
    squad_probs = np.array([p[1][3] for p in predictions])

    def _ensemble_logloss(weights_raw):
        """Compute log-loss for given weights (softmax to ensure sum=1, >0)."""
        exp_w = np.exp(weights_raw)
        w = exp_w / exp_w.sum()

        blended = (w[0] * sim_probs + w[1] * pois_probs +
                   w[2] * xgb_probs + w[3] * squad_probs)
        blended = np.clip(blended, 1e-6, 1.0)
        blended = blended / blended.sum(axis=1, keepdims=True)

        return -np.mean(np.log(blended[np.arange(len(actuals)), actuals]))

    best_result = None
    best_loss = np.inf

    starts = [
        np.array([0.0, 0.0, 0.0, 0.0]),  # equal weights
        np.array([0.3, 0.2, 0.0, -0.3]),  # favor sim/pois
        np.array([-0.2, -0.2, 0.3, 0.0]),  # favor xgb
        np.log(np.array(DEFAULT_WEIGHTS)),  # hand-picked as start
    ]

    for x0 in starts:
        result = minimize(_ensemble_logloss, x0, method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-6})
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result

    exp_w = np.exp(best_result.x)
    fitted_weights = tuple(exp_w / exp_w.sum())

    w = np.array(fitted_weights)
    blended = (w[0] * sim_probs + w[1] * pois_probs +
               w[2] * xgb_probs + w[3] * squad_probs)
    blended = np.clip(blended, 1e-6, 1.0)
    blended = blended / blended.sum(axis=1, keepdims=True)

    one_hot = np.zeros_like(blended)
    one_hot[np.arange(len(actuals)), actuals] = 1.0
    brier = np.mean(np.sum((blended - one_hot) ** 2, axis=1))

    ensemble_logloss = -np.mean(np.log(blended[np.arange(len(actuals)), actuals]))
    ensemble_preds = blended.argmax(axis=1)
    ensemble_acc = np.mean(ensemble_preds == actuals)

    cal_bins = _compute_calibration_bins(blended[:, 0], (actuals == 0).astype(float))

    backtest_results = {
        "n_matches": n_matches,
        "ensemble_logloss": ensemble_logloss,
        "ensemble_brier": brier,
        "ensemble_accuracy": ensemble_acc,
        "calibration_bins": cal_bins,
        "fitted_weights": fitted_weights,
    }

    return fitted_weights, backtest_results
