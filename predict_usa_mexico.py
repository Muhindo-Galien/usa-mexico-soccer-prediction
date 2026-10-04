"""USA vs Mexico — International Friendly, 2026-10-03, Glendale, AZ.

All the reusable modeling logic lives in engine/; all the player, match,
and bracket data lives in data/*.csv. This script just wires them
together for this one matchup and prints the report.

Both teams were eliminated in the Round of 16 of the 2026 World Cup by
other opponents (Mexico lost to England, USA lost to Belgium) — this is
their CONCACAF rematch a few months later. data/wc2026_results.csv now
carries the full tournament through the Semifinal, since all of it is
genuinely in the past relative to this match date.
"""

import os
import textwrap

import numpy as np

from engine.config import (
    TRAIN_START, VAL_START, ELO_BASE,
    N_SIMULATIONS, DIXON_COLES_RHO, MAX_GOALS,
)
from engine.historical import load_results, load_wc_patches, patch_results, normalize_country
from engine.elo_form import build_dataset, per_team_long, split_by_date
from engine.team_model import train_model, evaluate, predict_symmetric
from engine.players import (
    load_team_players, bench_players, compute_squad_strength, compute_bench_impact,
)
from engine.tournament import load_team_matches, compute_tournament_profile
from engine.simulation import compute_player_derived_lambdas, simulate_match
from engine.poisson_model import score_probability_matrix, extract_probabilities, top_scorelines
from engine.ensemble import fit_ensemble_weights, DEFAULT_WEIGHTS
from engine.methodology import build_methodology
from engine import charts, report

np.random.seed(42)

# ─── MATCH INFO ────────────────────────────────────────────────────────────
MATCH_DATE  = "2026-10-03"
HOME_TEAM   = "USA"
AWAY_TEAM   = "Mexico"
MATCH_STAGE = "International Friendly"
MATCH_VENUE = "State Farm Stadium, Glendale, AZ"
MATCH_CITY  = "Glendale"
HOME_COLOR, AWAY_COLOR = "#0A3161", "#006341"
HOME_FLAG, AWAY_FLAG = "🇺🇸", "🇲🇽"
REGRESSION_FACTOR = 1.00  # friendly, not a knockout game — no extra tightening

# ─── DATA FILES ─────────────────────────────────────────────────────────────
DATA_DIR = "data"
WC_PATCHES_CSV = os.path.join(DATA_DIR, "wc2026_results.csv")
PLAYERS_CSV = {HOME_TEAM: os.path.join(DATA_DIR, "players", "usa.csv"),
               AWAY_TEAM: os.path.join(DATA_DIR, "players", "mexico.csv")}
MATCHES_CSV = {HOME_TEAM: os.path.join(DATA_DIR, "matches", "usa.csv"),
               AWAY_TEAM: os.path.join(DATA_DIR, "matches", "mexico.csv")}

EXTRA_LIMITATIONS = [
    "**Friendly, not a knockout game** — there's less on the line than "
    "a World Cup match, both sides may rotate their squads, and the "
    "model has no 'rotation' or 'rivalry intensity' feature to capture "
    "that.",
    "**Months removed from the World Cup** — the player data here still "
    "reflects each team's summer 2026 tournament form; club form since "
    "then isn't re-checked automatically.",
]


def print_key_matchups(usa_squad, mex_squad):
    print(f"\n\n{'─' * 72}")
    print("  ⚔️  KEY PLAYER MATCHUPS")
    print(f"{'─' * 72}")
    for area, usa_desc, mex_desc in [
        ("STRIKERS", f"Downs (3G, PPI {usa_squad['player_ppis'][0][1]:.2f})",
         f"Gimenez (4G, PPI {mex_squad['player_ppis'][0][1]:.2f})"),
        ("CREATIVE HUBS", "Tillman (2G 1A) + Reyna (2G 2A)", "Pineda (1G 2A) + L. Chavez (0G 2A)"),
        ("WIDE THREAT", "Reyna (2G 2A)", "Lozano (2G 1A) — 76 caps, 18 career goals"),
        ("MIDFIELD ENGINE", "Adams (0G 1A) + Musah (0G 1A)", "Romo (0G 1A) + Lira (0G 0A)"),
        ("DEFENSIVE SPINE", "Richards + M. Robinson (yet to concede a set-piece goal)",
         "Vasquez + Reyes (0 goals conceded in 4 matches)"),
        ("GOALKEEPER", "Freese (2 CS, 76% saves)", "Rangel (4 CS, 84% saves — perfect record)"),
    ]:
        print(f"\n  {area}:")
        print(f"    {HOME_FLAG} {HOME_TEAM}: {usa_desc}")
        print(f"    {AWAY_FLAG} {AWAY_TEAM}: {mex_desc}")


def print_analysis_notes(prof_usa, prof_mex):
    print(f"\n\n{'─' * 72}")
    print("  📝 ANALYSIS NOTES")
    print(f"{'─' * 72}")
    paragraphs = [
        f"Both teams' 2026 World Cup runs ended in the Round of 16 — "
        f"Mexico fell 3-2 to England after going 4-0 through the group "
        f"stage and Round of 32, and USA lost 4-1 to Belgium despite "
        f"winning 3 of 4 on the way there (GD {prof_usa['GD']:+d} vs "
        f"Mexico's {prof_mex['GD']:+d}). Three months on, this CONCACAF "
        f"rematch in Glendale is a chance for both sides to reset the "
        f"narrative heading into the next World Cup cycle.",

        f"USA's attack leans on Damion Downs (3 goals in the group "
        f"stage and R32) finishing chances created by Malik Tillman "
        f"(2G 1A) and a finally-healthy Gio Reyna (2G 2A). Behind them, "
        f"the Adams-Musah double pivot protected a back line built "
        f"around Richards and Miles Robinson through most of the "
        f"tournament, before Belgium's attack finally got through it.",

        f"Mexico's case rests on a defense that didn't concede a goal "
        f"until the Round of 16 — Raul Rangel, Vasquez, Reyes, and the "
        f"128-cap Jesus Gallardo held firm through four straight clean "
        f"sheets. Up front, Santiago Gimenez was the tournament's "
        f"breakout scorer with 4 goals, with Hirving Lozano's experience "
        f"and Orbelin Pineda's creativity adding further scoring threat.",

        f"Mexico's underlying tournament numbers ({prof_mex['xG']:.2f} "
        f"xG total, {prof_mex['xGA/m']:.2f} xGA/match) were the "
        f"strongest of any team heading into the knockout rounds, while "
        f"USA's finishing ({prof_usa['Finishing']:.2f} G/xG ratio) "
        f"outpaced its process. With nothing riding on a friendly, "
        f"expect both coaches to rotate — but this is still a genuine "
        f"coin-flip battle in front of a Glendale crowd that will be "
        f"split down the middle.",
    ]
    for i, para in enumerate(paragraphs):
        if i:
            print()
        print(textwrap.fill(para, width=68, initial_indent="  ", subsequent_indent="  "))


def main():
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  ⚽ {HOME_FLAG} {HOME_TEAM.upper()} vs {AWAY_TEAM.upper()} {AWAY_FLAG} — PLAYER-LEVEL PREDICTION")
    print(f"  🏟️  {MATCH_STAGE} — CONCACAF rematch after the 2026 World Cup")
    print(f"  📅 {MATCH_DATE}  ·  📍 {MATCH_VENUE}")
    print(sep)

    # ── 0. Load data ─────────────────────────────────────────────────
    usa_players = load_team_players(PLAYERS_CSV[HOME_TEAM])
    mex_players = load_team_players(PLAYERS_CSV[AWAY_TEAM])
    usa_bench = bench_players(usa_players)
    mex_bench = bench_players(mex_players)
    usa_matches = load_team_matches(MATCHES_CSV[HOME_TEAM])
    mex_matches = load_team_matches(MATCHES_CSV[AWAY_TEAM])

    # ── 1. Player-level analysis ─────────────────────────────────────
    print(f"\n👟 [1/9] Computing Player Performance Indices ...")
    squad_usa = compute_squad_strength(usa_players)
    squad_mex = compute_squad_strength(mex_players)

    for label, flag, squad in ((HOME_TEAM, HOME_FLAG, squad_usa), (AWAY_TEAM, AWAY_FLAG, squad_mex)):
        print(f"\n  💪 {flag} {label} Squad Strength:")
        print(f"    Attack   : {squad['attack']:.3f}")
        print(f"    Midfield : {squad['midfield']:.3f}")
        print(f"    Defense  : {squad['defense']:.3f}")
        print(f"    GK       : {squad['gk']:.3f}")
        print(f"    Star bonus: {squad['star_bonus']:.3f}")
        print(f"    Overall  : {squad['overall']:.3f}")
        print(f"\n  ⭐ Top 5 Player Performance Indices — {label}:")
        for name, ppi in squad["player_ppis"][:5]:
            print(f"    {name:<24} PPI = {ppi:.3f}")

    # ── 2. Tournament profiles ───────────────────────────────────────
    print(f"\n📈 [2/9] Computing tournament profiles ...")
    prof_usa = compute_tournament_profile(usa_matches, HOME_TEAM)
    prof_mex = compute_tournament_profile(mex_matches, AWAY_TEAM)

    # ── 3. Player-derived expected goals ─────────────────────────────
    print(f"\n🎯 [3/9] Computing player-derived expected goals ...")
    lambda_usa, lambda_mex = compute_player_derived_lambdas(
        squad_usa, squad_mex, prof_usa, prof_mex,
        usa_bench, mex_bench, usa_matches, mex_matches,
        knockout_regression=REGRESSION_FACTOR)
    print(f"  {HOME_FLAG} {HOME_TEAM} lambda (player-adjusted) : {lambda_usa:.3f}")
    print(f"  {AWAY_FLAG} {AWAY_TEAM} lambda (player-adjusted) : {lambda_mex:.3f}")

    # ── 4. Monte Carlo simulation ────────────────────────────────────
    print(f"\n🎲 [4/9] Running Monte Carlo simulation ({N_SIMULATIONS:,} matches) ...")
    sim_result = simulate_match(lambda_usa, lambda_mex)
    p_sim_h, p_sim_d, p_sim_a = sim_result["p_home"], sim_result["p_draw"], sim_result["p_away"]
    print(f"  Simulation  ->  {HOME_FLAG} {HOME_TEAM} {p_sim_h*100:.1f}%  |  "
          f"🤝 Draw {p_sim_d*100:.1f}%  |  {AWAY_FLAG} {AWAY_TEAM} {p_sim_a*100:.1f}%")
    print(f"  ⚽ Avg goals ->  {HOME_TEAM} {sim_result['avg_goals_home']:.2f}  |  "
          f"{AWAY_TEAM} {sim_result['avg_goals_away']:.2f}")

    # ── 5. Dixon-Coles Poisson ───────────────────────────────────────
    print(f"\n🧮 [5/9] Building Dixon-Coles Poisson score matrix ...")
    matrix = score_probability_matrix(lambda_usa, lambda_mex, DIXON_COLES_RHO, MAX_GOALS)
    p_pois_h, p_pois_d, p_pois_a = extract_probabilities(matrix)
    print(f"  Poisson     ->  {HOME_FLAG} {HOME_TEAM} {p_pois_h*100:.1f}%  |  "
          f"🤝 Draw {p_pois_d*100:.1f}%  |  {AWAY_FLAG} {AWAY_TEAM} {p_pois_a*100:.1f}%")

    top_scores = top_scorelines(matrix, n=10)
    print(f"\n  🔢 Most likely scorelines:")
    for rank, (h_g, a_g, prob) in enumerate(top_scores, 1):
        outcome = (f"{HOME_TEAM} win" if h_g > a_g
                   else (f"{AWAY_TEAM} win" if a_g > h_g else "Draw"))
        print(f"    {rank:>2}. {HOME_TEAM} {h_g}-{a_g} {AWAY_TEAM}  "
              f"({prob*100:>5.2f}%)  [{outcome}]")

    # ── 6. Team-level XGBoost (as prior/anchor) ──────────────────────
    print(f"\n🤖 [6/9] Training team-level XGBoost model (anchor prior) ...")
    results = load_results()
    results = patch_results(results, load_wc_patches(WC_PATCHES_CSV))
    dataset, final_elo = build_dataset(results)
    long = per_team_long(results)

    # USA is stored as "United States" in the historical dataset (and in
    # the WC2026 patches) — translate just for the lookups that hit it.
    elo_home = normalize_country(HOME_TEAM)
    elo_away = normalize_country(AWAY_TEAM)

    h_elo = final_elo.get(elo_home, ELO_BASE)
    a_elo = final_elo.get(elo_away, ELO_BASE)
    print(f"  {HOME_FLAG} {HOME_TEAM} Elo  : {h_elo:.0f}")
    print(f"  {AWAY_FLAG} {AWAY_TEAM} Elo  : {a_elo:.0f}")
    print(f"  📏 Elo gap  : {h_elo - a_elo:+.0f}")

    train, val = split_by_date(dataset, TRAIN_START, VAL_START, MATCH_DATE)
    model, X_val, y_val = train_model(train, val)
    evaluate(model, X_val, y_val)

    p_xgb_h, p_xgb_d, p_xgb_a = predict_symmetric(
        model, long, final_elo, elo_home, elo_away, MATCH_DATE,
        neutral=True, weight=4)
    print(f"  XGBoost     ->  {HOME_FLAG} {HOME_TEAM} {p_xgb_h*100:.1f}%  |  "
          f"🤝 Draw {p_xgb_d*100:.1f}%  |  {AWAY_FLAG} {AWAY_TEAM} {p_xgb_a*100:.1f}%")

    # ── 7. Fit ensemble weights via backtesting ──────────────────────
    print(f"\n⚖️  [7/9] Fitting ensemble weights on historical knockout data ...")
    fitted_weights, backtest_results = fit_ensemble_weights(
        dataset, final_elo, long, model, knockout_regression=REGRESSION_FACTOR)
    W_SIM, W_POIS, W_XGB, W_SQUAD = fitted_weights

    print(f"\n  ⚖️  Fitted weights (log-loss minimized):")
    print(f"    Monte Carlo (player) : {W_SIM:.3f}  (hand-picked was {DEFAULT_WEIGHTS[0]:.2f})")
    print(f"    Dixon-Coles (player) : {W_POIS:.3f}  (hand-picked was {DEFAULT_WEIGHTS[1]:.2f})")
    print(f"    XGBoost (team)       : {W_XGB:.3f}  (hand-picked was {DEFAULT_WEIGHTS[2]:.2f})")
    print(f"    Squad strength       : {W_SQUAD:.3f}  (hand-picked was {DEFAULT_WEIGHTS[3]:.2f})")

    if backtest_results:
        print(f"\n  📊 Backtest performance ({backtest_results['n_matches']} major-tournament matches):")
        print(f"    Ensemble log-loss    : {backtest_results['ensemble_logloss']:.4f}")
        print(f"    Ensemble Brier score : {backtest_results['ensemble_brier']:.4f}")
        print(f"    Ensemble accuracy    : {backtest_results['ensemble_accuracy']:.3f}")
        if backtest_results.get("calibration_bins"):
            print(f"\n  📐 Calibration (P(home win) reliability):")
            print(f"    {'Predicted':>10} {'Actual':>10} {'N matches':>10}")
            for b in backtest_results["calibration_bins"]:
                print(f"    {b['predicted_mean']:>10.3f} {b['actual_mean']:>10.3f} {b['count']:>10d}")

    # ── 8. Ensemble blend ──────────────────────────────────────────────
    print(f"\n🔮 [8/9] Building ensemble prediction ...")
    str_diff = squad_usa["overall"] - squad_mex["overall"]
    p_squad_raw_h = 1.0 / (1.0 + np.exp(-str_diff * 3.0))
    p_squad_raw_a = 1.0 - p_squad_raw_h
    p_squad_d = 0.22
    p_squad_h = p_squad_raw_h * (1.0 - p_squad_d)
    p_squad_a = p_squad_raw_a * (1.0 - p_squad_d)

    print(f"  Squad model ->  {HOME_FLAG} {HOME_TEAM} {p_squad_h*100:.1f}%  |  "
          f"🤝 Draw {p_squad_d*100:.1f}%  |  {AWAY_FLAG} {AWAY_TEAM} {p_squad_a*100:.1f}%")
    print(f"\n  🔀 Ensemble weights (fitted): Simulation={W_SIM:.2f}  Poisson={W_POIS:.2f}  "
          f"XGBoost={W_XGB:.2f}  Squad={W_SQUAD:.2f}")

    ens_h = W_SIM * p_sim_h + W_POIS * p_pois_h + W_XGB * p_xgb_h + W_SQUAD * p_squad_h
    ens_d = W_SIM * p_sim_d + W_POIS * p_pois_d + W_XGB * p_xgb_d + W_SQUAD * p_squad_d
    ens_a = W_SIM * p_sim_a + W_POIS * p_pois_a + W_XGB * p_xgb_a + W_SQUAD * p_squad_a
    ens_tot = ens_h + ens_d + ens_a
    ens_h, ens_d, ens_a = ens_h / ens_tot, ens_d / ens_tot, ens_a / ens_tot

    print(f"\n  🏁 ENSEMBLE ->  {HOME_FLAG} {HOME_TEAM} {ens_h*100:.1f}%  |  "
          f"🤝 Draw {ens_d*100:.1f}%  |  {AWAY_FLAG} {AWAY_TEAM} {ens_a*100:.1f}%")

    # ── 9. Charts ────────────────────────────────────────────────────
    print(f"\n🎨 [9/9] Generating visualizations ...")
    out_dir = os.path.join("predictions", MATCH_DATE)
    os.makedirs(out_dir, exist_ok=True)

    chart_kwargs = dict(home_team=HOME_TEAM, away_team=AWAY_TEAM,
                         home_color=HOME_COLOR, away_color=AWAY_COLOR)
    chart_paths = [
        charts.make_prediction_chart(ens_h, ens_d, ens_a, out_dir, **chart_kwargs,
                                      match_stage=MATCH_STAGE, match_date=MATCH_DATE,
                                      city=MATCH_CITY),
        charts.make_player_ppi_chart(squad_usa, squad_mex, out_dir, **chart_kwargs),
        charts.make_position_comparison_chart(
            squad_usa, squad_mex, out_dir, **chart_kwargs,
            bench_h=usa_bench, bench_a=mex_bench, bench_impact_fn=compute_bench_impact),
        charts.make_score_heatmap(matrix, lambda_usa, lambda_mex, out_dir,
                                   home_team=HOME_TEAM, away_team=AWAY_TEAM,
                                   home_color=HOME_COLOR),
        charts.make_model_comparison_chart({
            "Monte Carlo\n(Player)": (p_sim_h, p_sim_d, p_sim_a),
            "Dixon-Coles\n(Player)": (p_pois_h, p_pois_d, p_pois_a),
            "XGBoost\n(Team)": (p_xgb_h, p_xgb_d, p_xgb_a),
            "Squad\nStrength": (p_squad_h, p_squad_d, p_squad_a),
            "ENSEMBLE": (ens_h, ens_d, ens_a),
        }, out_dir, **chart_kwargs),
    ]

    # ══════════════════════════════════════════════════════════════════
    #  COMPREHENSIVE REPORT
    # ══════════════════════════════════════════════════════════════════
    report.print_final_prediction(HOME_TEAM, AWAY_TEAM, MATCH_STAGE, ens_h, ens_d, ens_a,
                                   lambda_usa, lambda_mex, top_scores)
    report.print_model_comparison(HOME_TEAM, AWAY_TEAM, [
        ("Monte Carlo (player-derived)", p_sim_h, p_sim_d, p_sim_a),
        ("Dixon-Coles Poisson (player)", p_pois_h, p_pois_d, p_pois_a),
        ("XGBoost (team-level)", p_xgb_h, p_xgb_d, p_xgb_a),
        ("Squad strength comparison", p_squad_h, p_squad_d, p_squad_a),
        ("ENSEMBLE (weighted blend)", ens_h, ens_d, ens_a),
    ])
    report.print_squad_breakdown(HOME_TEAM, AWAY_TEAM, squad_usa, squad_mex)
    report.print_tournament_comparison(HOME_TEAM, AWAY_TEAM, prof_usa, prof_mex)
    print_key_matchups(squad_usa, squad_mex)
    report.print_match_log(HOME_TEAM, usa_matches)
    report.print_match_log(AWAY_TEAM, mex_matches)
    report.print_bench_impact(HOME_TEAM, AWAY_TEAM, usa_bench, mex_bench)
    print_analysis_notes(prof_usa, prof_mex)
    report.print_head_to_head(long, elo_home, elo_away, HOME_TEAM, AWAY_TEAM, MATCH_DATE)
    report.print_charts_saved(MATCH_DATE, chart_paths)

    methodology = build_methodology(
        fitted_weights, DEFAULT_WEIGHTS, backtest_results,
        knockout_regression=REGRESSION_FACTOR,
        extra_limitations=EXTRA_LIMITATIONS)
    print(f"\n\n{'═' * 72}")
    print("  📖 METHODOLOGY & LIMITATIONS")
    print(f"{'═' * 72}")
    print(methodology)

    methodology_path = os.path.join(out_dir, "METHODOLOGY.md")
    with open(methodology_path, "w") as f:
        f.write(methodology)
    print(f"\n  💾 Saved to {methodology_path}")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    main()
