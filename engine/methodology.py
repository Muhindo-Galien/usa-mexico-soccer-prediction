"""Builds the generic methodology/limitations writeup for a prediction
run. `extra_limitations` lets a matchup script append notes specific to
its own setup (e.g. a hypothetical bracket slot)."""

DEFAULT_WEIGHTS = (0.35, 0.30, 0.20, 0.15)


def build_methodology(fitted_weights, hand_picked, backtest_results,
                       knockout_regression=0.95, extra_limitations=None):
    lines = []
    lines.append("# Methodology & Limitations\n")
    lines.append("## Model Architecture\n")
    lines.append("Hybrid ensemble combining four component models:\n")
    lines.append("1. **Monte Carlo simulation** (10,000 Poisson-distributed matches)")
    lines.append("   - Lambdas derived from player-level PPI position-group strengths")
    lines.append("   - Blended 40% player-ratio + 60% tournament-xG-ratio")
    lines.append("2. **Dixon-Coles Poisson** score matrix (same player-derived lambdas)")
    lines.append("   - Low-score correlation correction (rho parameter)")
    lines.append("3. **XGBoost multi-class classifier** (team-level historical prior)")
    lines.append("   - Trained on 20 years of international results (2006-present)")
    lines.append("   - Features: Elo ratings, 5/10-match form, head-to-head, rest days")
    lines.append("4. **Squad strength comparison** (PPI-based sigmoid)\n")

    lines.append("## What is fitted from data\n")
    lines.append(f"- **Ensemble blend weights**: fitted via scipy.optimize on "
                 f"major-tournament matches since 2010 (minimizing log-loss)")
    lines.append(f"  - Fitted: Sim={fitted_weights[0]:.3f}, Pois={fitted_weights[1]:.3f}, "
                 f"XGB={fitted_weights[2]:.3f}, Squad={fitted_weights[3]:.3f}")
    lines.append(f"  - Hand-picked comparison: Sim={hand_picked[0]:.2f}, Pois={hand_picked[1]:.2f}, "
                 f"XGB={hand_picked[2]:.2f}, Squad={hand_picked[3]:.2f}")
    lines.append("- **XGBoost model**: hyperparameters tuned via early stopping on "
                 "the validation window")
    lines.append("- **Elo ratings**: fitted iteratively on all historical results")
    lines.append("- **Momentum feature**: computed from actual knockout GD (tanh-normalized)\n")

    if backtest_results:
        lines.append("## Ensemble calibration (walk-forward backtest)\n")
        lines.append(f"- Backtest set: {backtest_results['n_matches']} major-tournament "
                     f"matches (2010-2025)")
        lines.append(f"- Log-loss: {backtest_results['ensemble_logloss']:.4f}")
        lines.append(f"- Brier score (multi-class): {backtest_results['ensemble_brier']:.4f}")
        lines.append(f"- Accuracy: {backtest_results['ensemble_accuracy']:.3f}")
        if backtest_results.get("calibration_bins"):
            lines.append(f"\n  Reliability (P(home win)):")
            lines.append(f"  | Predicted | Actual | N |")
            lines.append(f"  |-----------|--------|---|")
            for b in backtest_results["calibration_bins"]:
                lines.append(f"  | {b['predicted_mean']:.3f} | {b['actual_mean']:.3f} "
                             f"| {b['count']} |")
        lines.append("")

    lines.append("## What remains a manual choice\n")
    lines.append("- **Starting XI selection**: 11 players per squad, picked from each "
                 "team's full roster by caps/role — not the whole squad list")
    lines.append("- **PPI formula weights**: position-aware sub-weights (30% club / 70% WC)")
    lines.append("  - GK: save_pct 0.35, CS_rate 0.30, GA_rate 0.35")
    lines.append("  - CB/FB: tackles 0.15, interceptions 0.15, aerials 0.08, goals 0.15, assists 0.10")
    lines.append("  - CM: tackles 0.10, interceptions 0.10, key_passes 0.10, goals 0.20, assists 0.15")
    lines.append("  - AM/W/ST: goals 0.25-0.30, assists 0.15, xG/SoT/dribbles smaller weights")
    lines.append("- **Dixon-Coles rho = -0.13** (low-score correlation correction)")
    lines.append("- **Position-group aggregation**: 35% ATT + 25% MID + 25% DEF + 15% GK")
    lines.append("- **Star bonus**: top 3 players get 0.15/0.12/0.09 uplift weights")
    lines.append(f"- **Regression factor = {knockout_regression:.2f}** "
                 f"(scales expected goals for how much is riding on this fixture)")
    lines.append("- **Tournament average xG = 1.40** (assumed baseline)")
    lines.append("- **Bench impact formula**: club_rating/7 * 0.4 + "
                 "(goals*0.3 + assists*0.15) * 0.6, scaled by impact tier\n")

    lines.append("## Key limitations\n")
    lines.append("1. **PPI weights are not fitted** — position sub-weights are hand-tuned "
                 "heuristics, not optimized against outcomes.")
    lines.append("2. **No player-level data in backtest** — the ensemble weight fitting uses "
                 "Elo-based proxies for the Monte Carlo/Dixon-Coles components (no per-player "
                 "stats available in results.csv for historical matches).")
    lines.append("3. **Dixon-Coles rho is assumed** — not fitted from this tournament's data "
                 "(would need >50 matches to estimate reliably).")
    lines.append("4. **Player data is point-in-time** — club/WC stats are manually "
                 "entered in data/players/*.csv, not from a live data feed.")
    lines.append("5. **XGBoost sees team entities, not personnel** — a squad overhaul "
                 "is invisible to the team-level model.")

    n = 6
    for note in (extra_limitations or []):
        lines.append(f"{n}. {note}")
        n += 1
    lines.append("")

    return "\n".join(lines)
