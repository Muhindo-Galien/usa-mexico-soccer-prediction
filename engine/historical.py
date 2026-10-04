"""Loading + patching the historical international-results dataset."""

import os
import pandas as pd
import requests

from .config import CACHE_DIR, RESULTS_URL, NAME_MAP


def fetch_results():
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "results.csv")
    if not os.path.exists(path):
        resp = requests.get(RESULTS_URL, timeout=120)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
    return pd.read_csv(path)


def normalize_country(name):
    return NAME_MAP.get(name, name) if isinstance(name, str) else name


def load_results():
    r = fetch_results()
    r["home_team"] = r["home_team"].map(normalize_country)
    r["away_team"] = r["away_team"].map(normalize_country)
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna(subset=["home_score", "away_score"]).copy()
    r["home_score"] = r["home_score"].astype(int)
    r["away_score"] = r["away_score"].astype(int)
    r["neutral"] = r["neutral"].astype(str).str.upper().eq("TRUE").astype(int)
    return r.sort_values("date").reset_index(drop=True)


def load_wc_patches(csv_path):
    """Read a WC-bracket patch file (date,home,away,home_score,away_score[,stage])
    into the {(date, home, away): (home_score, away_score)} form patch_results()
    expects."""
    p = pd.read_csv(csv_path)
    patches = {}
    for _, row in p.iterrows():
        patches[(row["date"], row["home"], row["away"])] = (int(row["home_score"]), int(row["away_score"]))
    return patches


def patch_results(r, patches):
    """Apply a dict of {(date, home, away): (home_score, away_score)} patches
    on top of the real historical results — updating a match if it already
    exists, or adding it as a new (neutral-venue) row otherwise."""
    patched, added = 0, 0
    for (date_str, home, away), (hs, as_) in patches.items():
        dt = pd.Timestamp(date_str)
        mask = (r["date"] == dt) & (r["home_team"] == home) & (r["away_team"] == away)
        if mask.any():
            idx = r.index[mask][0]
            if pd.isna(r.at[idx, "home_score"]) or r.at[idx, "home_score"] != hs:
                r.at[idx, "home_score"] = hs
                r.at[idx, "away_score"] = as_
                patched += 1
        else:
            new = pd.DataFrame([{
                "date": dt, "home_team": home, "away_team": away,
                "home_score": hs, "away_score": as_,
                "tournament": "FIFA World Cup", "city": "USA",
                "country": "United States", "neutral": 1,
            }])
            r = pd.concat([r, new], ignore_index=True)
            added += 1
    r["home_score"] = r["home_score"].astype(int)
    r["away_score"] = r["away_score"].astype(int)
    r = r.sort_values("date").reset_index(drop=True)
    print(f"  🩹 Patch applied: {patched} updated, {added} added")
    return r


def tournament_weight(name):
    t = str(name).lower()
    if "fifa world cup" in t and "qualif" not in t: return 4
    if "qualif" in t: return 3
    big = ["uefa nations", "copa america", "afc asian cup", "africa cup",
           "concacaf", "uefa euro", "confederations"]
    if any(tok in t for tok in big): return 3
    if "friendly" in t: return 1
    return 2
