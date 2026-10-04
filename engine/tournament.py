"""A team's match-by-match tournament record (data/matches/<team>.csv)
and the aggregate profile derived from it."""

import numpy as np
import pandas as pd

from .config import STAGE_WEIGHT

_INT_COLS = ["gf", "ga", "shots", "sot", "poss"]
_FLOAT_COLS = ["xg", "xga"]


def load_team_matches(csv_path):
    df = pd.read_csv(csv_path)
    matches = []
    for _, row in df.iterrows():
        m = {}
        for col, val in row.items():
            if pd.isna(val):
                continue
            if col in _INT_COLS:
                m[col] = int(val)
            elif col in _FLOAT_COLS:
                m[col] = float(val)
            else:
                m[col] = val
        matches.append(m)
    return matches


def compute_tournament_profile(matches, team_name, stage_weight=STAGE_WEIGHT):
    n = len(matches)
    gf   = sum(m["gf"] for m in matches)
    ga   = sum(m["ga"] for m in matches)
    xg   = sum(m["xg"] for m in matches)
    xga  = sum(m["xga"] for m in matches)
    sot  = sum(m["sot"] for m in matches)
    shots = sum(m["shots"] for m in matches)
    poss = np.mean([m["poss"] for m in matches])
    cs   = sum(1 for m in matches if m["ga"] == 0)
    wins = sum(1 for m in matches if m["result"] == "W")
    draws = sum(1 for m in matches if m["result"] == "D")

    # Stage-weighted xG (later stages count more)
    total_w = sum(stage_weight[m["stage"]] for m in matches)
    xg_w  = sum(m["xg"]  * stage_weight[m["stage"]] for m in matches) / total_w
    xga_w = sum(m["xga"] * stage_weight[m["stage"]] for m in matches) / total_w

    return {
        "team": team_name, "played": n,
        "W": wins, "D": draws, "L": n - wins - draws,
        "GF": gf, "GA": ga, "GD": gf - ga,
        "GF/m": gf / n, "GA/m": ga / n,
        "xG": xg, "xGA": xga, "xGD": xg - xga,
        "xG/m": xg / n, "xGA/m": xga / n,
        "xG/m_w": xg_w, "xGA/m_w": xga_w,
        "Shots": shots, "SoT": sot,
        "SoT%": 100 * sot / max(shots, 1),
        "Poss%": poss,
        "CS": cs, "CS%": 100 * cs / n,
        "Win%": 100 * (wins + 0.5 * draws) / n,
        "Finishing": gf / max(xg, 0.01),
        "Def_solidity": xga / max(ga, 0.01),
    }
