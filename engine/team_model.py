"""Team-level XGBoost classifier: trained on historical international
results, used as one of the ensemble's component models."""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

from .config import TEAM_FEATURES, ELO_BASE
from .elo_form import form_as_of, h2h_as_of


def train_model(train, val):
    X_train, y_train = train[TEAM_FEATURES].astype(float), train["label"].astype(int)
    X_val,   y_val   = val[TEAM_FEATURES].astype(float),   val["label"].astype(int)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=600,
        learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=1.0, eval_metric="mlogloss", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, X_val, y_val


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred  = proba.argmax(axis=1)
    base  = np.tile(np.bincount(y_val, minlength=3) / len(y_val), (len(y_val), 1))
    acc   = accuracy_score(y_val, pred)
    ll    = log_loss(y_val, proba)
    bl    = log_loss(y_val, base, labels=[0, 1, 2])
    print(f"  ✅ Validation accuracy : {acc:.3f}")
    print(f"  📉 Validation log-loss : {ll:.3f}  (baseline {bl:.3f})")
    return acc, ll


def build_match_row(long, final_elo, home, away, neutral, weight, asof):
    hf, af = form_as_of(long, home, asof), form_as_of(long, away, asof)
    he, ae = final_elo.get(home, ELO_BASE), final_elo.get(away, ELO_BASE)
    n, wr, gd = h2h_as_of(long, home, away, asof)
    row = {"neutral": int(neutral), "tournament_weight": weight,
           "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
           "home_win5": hf["win5"], "away_win5": af["win5"],
           "home_gd5": hf["gd5"], "away_gd5": af["gd5"],
           "home_win10": hf["win10"], "away_win10": af["win10"],
           "home_rest_days": hf["rest_days"], "away_rest_days": af["rest_days"],
           "h2h_n": n, "h2h_home_winrate": wr, "h2h_home_gd": gd}
    return pd.DataFrame([row])[TEAM_FEATURES].astype(float)


def predict_symmetric(model, long, final_elo, a, b, asof, neutral, weight):
    """Average the model's (a vs b) and (b vs a) predictions so the result
    doesn't depend on which team is passed as "home"."""
    p_ab = model.predict_proba(
        build_match_row(long, final_elo, a, b, neutral, weight, asof))[0]
    p_ba = model.predict_proba(
        build_match_row(long, final_elo, b, a, neutral, weight, asof))[0]
    p_a = (p_ab[0] + p_ba[2]) / 2.0
    p_d = (p_ab[1] + p_ba[1]) / 2.0
    p_b = (p_ab[2] + p_ba[0]) / 2.0
    tot = p_a + p_d + p_b
    return p_a / tot, p_d / tot, p_b / tot
