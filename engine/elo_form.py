"""Elo ratings, rolling form, and head-to-head features over the
historical results dataset."""

import numpy as np
import pandas as pd

from .config import ELO_BASE, ELO_K, ELO_HOME_BONUS
from .historical import tournament_weight


def add_label_and_context(r):
    r = r.copy()
    r["label"] = np.where(r["home_score"] > r["away_score"], 0,
                          np.where(r["home_score"] == r["away_score"], 1, 2))
    r["tournament_weight"] = r["tournament"].map(tournament_weight)
    return r


def compute_elo(r):
    r = r.sort_values("date").reset_index(drop=True)
    rating, home_pre, away_pre = {}, np.zeros(len(r)), np.zeros(len(r))
    for i, row in r.iterrows():
        rh = rating.get(row.home_team, ELO_BASE)
        ra = rating.get(row.away_team, ELO_BASE)
        home_pre[i], away_pre[i] = rh, ra
        bonus = 0 if row.neutral == 1 else ELO_HOME_BONUS
        exp_home = 1 / (1 + 10 ** (-((rh + bonus) - ra) / 400))
        score_home = 1.0 if row.label == 0 else (0.5 if row.label == 1 else 0.0)
        margin = abs(int(row.home_score) - int(row.away_score))
        mult = np.log(max(margin, 1) + 1) * (2.2 / (abs(rh - ra) * 0.001 + 2.2))
        rating[row.home_team] = rh + ELO_K * mult * (score_home - exp_home)
        rating[row.away_team] = ra + ELO_K * mult * ((1 - score_home) - (1 - exp_home))
    r["home_elo"], r["away_elo"] = home_pre, away_pre
    r["elo_diff"] = home_pre - away_pre
    return r, rating


def per_team_long(r):
    home = pd.DataFrame({"date": r["date"].values, "team": r["home_team"].values,
                         "opp": r["away_team"].values, "gf": r["home_score"].values,
                         "ga": r["away_score"].values})
    away = pd.DataFrame({"date": r["date"].values, "team": r["away_team"].values,
                         "opp": r["home_team"].values, "gf": r["away_score"].values,
                         "ga": r["home_score"].values})
    long = pd.concat([home, away], ignore_index=True)
    long["result"] = np.where(long["gf"] > long["ga"], 1.0,
                              np.where(long["gf"] == long["ga"], 0.5, 0.0))
    long["gd"] = long["gf"] - long["ga"]
    return long


def add_form_features(r):
    long = per_team_long(r).sort_values(["team", "date"]).reset_index(drop=True)
    long["prev_date"]  = long.groupby("team")["date"].shift(1)
    long["result_lag"] = long.groupby("team")["result"].shift(1)
    long["gd_lag"]     = long.groupby("team")["gd"].shift(1)
    long["win5"]  = long.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    long["gd5"]   = long.groupby("team")["gd_lag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    long["win10"] = long.groupby("team")["result_lag"].transform(
        lambda s: s.rolling(10, min_periods=1).mean())
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days
    form = long[["date", "team", "win5", "gd5", "win10", "rest_days"]].drop_duplicates(["date", "team"])
    r = r.merge(form.rename(columns={"team": "home_team", "win5": "home_win5", "gd5": "home_gd5",
                                     "win10": "home_win10", "rest_days": "home_rest_days"}),
                on=["date", "home_team"], how="left")
    r = r.merge(form.rename(columns={"team": "away_team", "win5": "away_win5", "gd5": "away_gd5",
                                     "win10": "away_win10", "rest_days": "away_rest_days"}),
                on=["date", "away_team"], how="left")
    return r


def add_h2h_features(r):
    long = per_team_long(r).sort_values(["team", "opp", "date"]).reset_index(drop=True)
    g = long.groupby(["team", "opp"])
    long["h2h_n"]       = g.cumcount()
    long["h2h_winrate"] = g["result"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    long["h2h_gd"]      = g["gd"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    h2h = long[["date", "team", "opp", "h2h_n", "h2h_winrate", "h2h_gd"]].drop_duplicates(
        ["date", "team", "opp"])
    r = r.merge(h2h.rename(columns={"team": "home_team", "opp": "away_team",
                                    "h2h_winrate": "h2h_home_winrate", "h2h_gd": "h2h_home_gd"}),
                on=["date", "home_team", "away_team"], how="left")
    return r


def build_dataset(r):
    r = add_label_and_context(r)
    r, final_elo = compute_elo(r)
    r = add_form_features(r)
    r = add_h2h_features(r)
    return r, final_elo


def split_by_date(ds, train_start, val_start, cutoff):
    train = ds[(ds["date"] >= pd.Timestamp(train_start))
               & (ds["date"] < pd.Timestamp(val_start))].copy()
    val   = ds[(ds["date"] >= pd.Timestamp(val_start))
               & (ds["date"] < pd.Timestamp(cutoff))].copy()
    return train, val


def form_as_of(long, team, asof):
    sub = long[(long["team"] == team) & (long["date"] < pd.Timestamp(asof))].sort_values("date")
    if len(sub) == 0:
        return {"win5": 0.5, "gd5": 0.0, "win10": 0.5, "rest_days": 30.0}
    l5, l10 = sub.tail(5), sub.tail(10)
    return {"win5": float(l5["result"].mean()), "gd5": float((l5["gf"] - l5["ga"]).mean()),
            "win10": float(l10["result"].mean()),
            "rest_days": float((pd.Timestamp(asof) - sub["date"].max()).days)}


def h2h_as_of(long, team, opp, asof):
    sub = long[(long["team"] == team) & (long["opp"] == opp) & (long["date"] < pd.Timestamp(asof))]
    if len(sub) == 0:
        return 0.0, np.nan, np.nan
    return float(len(sub)), float(sub["result"].mean()), float(sub["gd"].mean())
