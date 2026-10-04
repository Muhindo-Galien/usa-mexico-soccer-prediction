"""Player-level data loading and the Player Performance Index (PPI) model.

Each team's full squad lives in a CSV (data/players/<team>.csv) with one
row per player and a `role` column of "starter" or "bench". Starters feed
compute_squad_strength (position-group PPI aggregation); bench players
feed compute_bench_impact (a simpler substitute-impact score).
"""

import numpy as np
import pandas as pd

_INT_COLS = [
    "age", "club_apps", "club_goals", "club_assists", "club_cs",
    "wc_apps", "wc_mins", "wc_goals", "wc_assists", "wc_tackles",
    "wc_interceptions", "wc_aerials_won", "wc_key_passes", "wc_dribbles",
    "wc_shots", "wc_sot", "wc_saves",
]
_FLOAT_COLS = ["club_rating", "club_ga_90", "wc_xg", "wc_save_pct"]


def load_team_players(csv_path):
    """Load a team's full squad (starters + bench) from CSV into a list of
    dicts. Blank cells are simply omitted from the dict so downstream
    `player.get(key, default)` calls fall back correctly."""
    df = pd.read_csv(csv_path)
    players = []
    for _, row in df.iterrows():
        p = {}
        for col, val in row.items():
            if pd.isna(val):
                continue
            if col in _INT_COLS:
                p[col] = int(val)
            elif col in _FLOAT_COLS:
                p[col] = float(val)
            else:
                p[col] = val
        p["starter"] = (p.get("role") == "starter")
        players.append(p)
    return players


def bench_players(players):
    return [p for p in players if p.get("role") == "bench"]


def compute_player_performance_index(player, team_matches=None):
    """
    Composite Player Performance Index (PPI), position-aware:
      - Attackers (ST, W):   goals, xG, shots on target, dribbles weighted heavily
      - Midfielders (AM, CM): key passes, assists, tackles, goals balanced
      - Defenders (CB, FB):  tackles, interceptions, aerials, clean sheets
      - Goalkeepers (GK):    save %, clean sheets, goals against

    Also factors in club season form as a baseline quality indicator.
    `team_matches` is accepted for forward-compatibility (future versions
    may normalize against opponent strength) but isn't used yet.
    """
    pos = player["pos"]

    # ── Club season baseline (30% of total PPI) ──
    club_rating = player.get("club_rating", 6.5)
    club_goals = player.get("club_goals", 0)
    club_assists = player.get("club_assists", 0)
    club_apps = player.get("club_apps", 1)

    if pos == "GK":
        club_cs = player.get("club_cs", 0)
        club_ga_90 = player.get("club_ga_90", 1.2)
        club_score = (club_rating / 7.0) * 0.4 + (club_cs / max(club_apps, 1)) * 0.3 + max(0, (1.5 - club_ga_90)) * 0.3
    elif pos in ("CB", "FB"):
        club_score = (club_rating / 7.0) * 0.5 + (club_goals + club_assists) / max(club_apps, 1) * 2.0 * 0.5
    elif pos in ("CM", "AM"):
        club_score = (club_rating / 7.0) * 0.4 + (club_goals + club_assists) / max(club_apps, 1) * 1.5 * 0.6
    else:  # ST, W
        club_score = (club_rating / 7.0) * 0.3 + (club_goals + club_assists) / max(club_apps, 1) * 1.2 * 0.7

    # ── World Cup tournament form (70% of total PPI) ──
    wc_mins = player.get("wc_mins", 0)
    if wc_mins == 0:
        return club_score * 0.3

    per_90 = 90.0 / max(wc_mins, 90)

    if pos == "GK":
        wc_save_pct = player.get("wc_save_pct", 0.65)
        wc_cs = player.get("wc_cs", 0)
        wc_ga = player.get("wc_ga", 0)
        wc_apps = player.get("wc_apps", 1)
        wc_score = (
            wc_save_pct * 0.35 +
            (wc_cs / max(wc_apps, 1)) * 0.30 +
            max(0, (1.5 - wc_ga / max(wc_apps, 1))) / 1.5 * 0.35
        )
    elif pos in ("CB", "FB"):
        tackles = player.get("wc_tackles", 0) * per_90
        interceptions = player.get("wc_interceptions", 0) * per_90
        aerials = player.get("wc_aerials_won", 0) * per_90
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        wc_score = (
            min(tackles * 0.15, 0.35) +
            min(interceptions * 0.15, 0.25) +
            min(aerials * 0.08, 0.15) +
            goals * 0.15 +
            assists * 0.10
        )
    elif pos == "CM":
        tackles = player.get("wc_tackles", 0) * per_90
        interceptions = player.get("wc_interceptions", 0) * per_90
        key_passes = player.get("wc_key_passes", 0) * per_90
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        wc_score = (
            min(tackles * 0.10, 0.20) +
            min(interceptions * 0.10, 0.15) +
            min(key_passes * 0.10, 0.20) +
            goals * 0.20 +
            assists * 0.15 +
            0.10  # base for availability
        )
    elif pos == "AM":
        key_passes = player.get("wc_key_passes", 0) * per_90
        dribbles = player.get("wc_dribbles", 0) * per_90
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        xg = player.get("wc_xg", 0) * per_90
        wc_score = (
            min(key_passes * 0.10, 0.20) +
            min(dribbles * 0.05, 0.15) +
            goals * 0.25 +
            assists * 0.15 +
            min(xg * 0.10, 0.15) +
            0.10
        )
    else:  # ST, W — attackers
        goals = player.get("wc_goals", 0)
        assists = player.get("wc_assists", 0)
        xg = player.get("wc_xg", 0)
        sot = player.get("wc_sot", 0) * per_90
        dribbles = player.get("wc_dribbles", 0) * per_90
        wc_score = (
            goals * 0.30 +
            assists * 0.15 +
            min(sot * 0.05, 0.15) +
            min(dribbles * 0.03, 0.10) +
            min(xg * 0.05, 0.15) +
            0.15  # base for participation
        )

    return club_score * 0.30 + wc_score * 0.70


def compute_squad_strength(players, team_matches=None):
    """
    Compute position-group strength indices from player PPIs.
    Returns a dict with attack, midfield, defense, gk, and overall scores.
    """
    starters = [p for p in players if p.get("starter", False)]

    gk_ppi = [compute_player_performance_index(p, team_matches)
              for p in starters if p["pos"] == "GK"]
    def_ppi = [compute_player_performance_index(p, team_matches)
               for p in starters if p["pos"] in ("CB", "FB")]
    mid_ppi = [compute_player_performance_index(p, team_matches)
               for p in starters if p["pos"] in ("CM", "AM")]
    att_ppi = [compute_player_performance_index(p, team_matches)
               for p in starters if p["pos"] in ("ST", "W")]

    gk_strength  = np.mean(gk_ppi) if gk_ppi else 0.5
    def_strength = np.mean(def_ppi) if def_ppi else 0.5
    mid_strength = np.mean(mid_ppi) if mid_ppi else 0.5
    att_strength = np.mean(att_ppi) if att_ppi else 0.5

    # Star player bonus: top performers lift entire team
    all_ppi = [(p["name"], compute_player_performance_index(p, team_matches))
               for p in starters]
    all_ppi.sort(key=lambda x: x[1], reverse=True)
    star_bonus = 0.0
    for i, (name, ppi) in enumerate(all_ppi[:3]):  # top 3 players
        star_bonus += ppi * (0.15 - i * 0.03)  # diminishing weight

    overall = (att_strength * 0.35 + mid_strength * 0.25 +
               def_strength * 0.25 + gk_strength * 0.15 + star_bonus)

    return {
        "gk": gk_strength,
        "defense": def_strength,
        "midfield": mid_strength,
        "attack": att_strength,
        "star_bonus": star_bonus,
        "overall": overall,
        "player_ppis": all_ppi,
    }


def compute_bench_impact(bench):
    """Estimate potential impact from substitutes."""
    total = 0.0
    for p in bench:
        base = p.get("club_rating", 6.5) / 7.0
        goals = p.get("wc_goals", 0)
        assists = p.get("wc_assists", 0)
        impact_mult = {"high": 1.3, "medium": 1.0, "low": 0.7}.get(p.get("impact", "medium"), 1.0)
        total += (base * 0.4 + (goals * 0.3 + assists * 0.15) * 0.6) * impact_mult
    return total / max(len(bench), 1)
