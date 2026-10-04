"""Generic, matchup-agnostic report sections. Everything here just prints
data that was already computed elsewhere — team/player names are passed
in as plain arguments, nothing is hardcoded to a specific matchup."""

import pandas as pd

from .elo_form import h2h_as_of
from .players import compute_bench_impact

STRENGTH_EMOJI = {"STRONG LEAN": "🔒", "LEAN": "👍", "TOSS-UP": "🤷"}


def print_final_prediction(home_team, away_team, match_stage, ens_h, ens_d, ens_a,
                            lambda_h, lambda_a, top_scores):
    sep = "=" * 72
    outcomes = [(home_team, ens_h), ("Draw", ens_d), (away_team, ens_a)]
    pick, conf = max(outcomes, key=lambda x: x[1])
    strength = "STRONG LEAN" if conf >= 0.55 else ("LEAN" if conf >= 0.42 else "TOSS-UP")

    print(f"\n\n{sep}")
    print(f"  🏆 FINAL PREDICTION — {match_stage.upper()}")
    print(f"  🔮 PLAYER-LEVEL ENSEMBLE MODEL")
    print(sep)
    print(f"\n  ⚽ 90-minute probabilities (ensemble):")
    print(f"    {home_team:<22} win   {ens_h*100:>5.1f}%")
    print(f"    {'🤝 Draw':<22}       {ens_d*100:>5.1f}%")
    print(f"    {away_team:<22} win   {ens_a*100:>5.1f}%")
    print(f"\n  🥅 Expected goals:")
    print(f"    {home_team:<22} xG   {lambda_h:>5.2f}")
    print(f"    {away_team:<22} xG   {lambda_a:>5.2f}")
    print(f"    Most likely score: {home_team} {top_scores[0][0]}-"
          f"{top_scores[0][1]} {away_team}  ({top_scores[0][2]*100:.1f}%)")
    print(f"\n  {'-' * 62}")
    print(f"  {STRENGTH_EMOJI.get(strength, '🎯')} PICK: {pick}  ({conf*100:.1f}%)   [{strength}]")
    print(sep)
    return pick, conf, strength


def print_model_comparison(home_team, away_team, models):
    """models: list of (name, p_home, p_draw, p_away) tuples."""
    print(f"\n\n{'─' * 72}")
    print("  📊 MODEL COMPARISON")
    print(f"{'─' * 72}")
    print(f"  {'Model':<32} {home_team:>10} {'Draw':>10} {away_team:>10}")
    print(f"  {'─'*32} {'─'*10} {'─'*10} {'─'*10}")
    for name, h, d, a in models:
        print(f"  {name:<32} {h*100:>9.1f}% {d*100:>9.1f}% {a*100:>9.1f}%")


def print_squad_breakdown(home_team, away_team, squad_h, squad_a):
    print(f"\n\n{'─' * 72}")
    print("  💪 SQUAD STRENGTH BREAKDOWN (Player Performance Index)")
    print(f"{'─' * 72}")
    print(f"  {'Position Group':<20} {home_team:>12} {away_team:>12} {'Edge':>12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")
    for name, hv, av in [
        ("Attack",   squad_h["attack"],   squad_a["attack"]),
        ("Midfield", squad_h["midfield"], squad_a["midfield"]),
        ("Defense",  squad_h["defense"],  squad_a["defense"]),
        ("Goalkeeper", squad_h["gk"],     squad_a["gk"]),
        ("Star Bonus", squad_h["star_bonus"], squad_a["star_bonus"]),
        ("OVERALL",  squad_h["overall"],  squad_a["overall"]),
    ]:
        edge = hv - av
        winner = home_team if edge > 0 else (away_team if edge < 0 else "Even")
        print(f"  {name:<20} {hv:>12.3f} {av:>12.3f} {edge:>+11.3f} {winner}")


def print_tournament_comparison(home_team, away_team, prof_h, prof_a):
    n = prof_h["played"]
    print(f"\n\n{'─' * 72}")
    print(f"  📅 TOURNAMENT PERFORMANCE COMPARISON ({n} matches each)")
    print(f"{'─' * 72}")
    print(f"  {'Metric':<28} {home_team:>12} {away_team:>12} {'Edge':>12}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*12}")
    rows = [
        ("Record (W-D-L)", f"{prof_h['W']}-{prof_h['D']}-{prof_h['L']}",
         f"{prof_a['W']}-{prof_a['D']}-{prof_a['L']}", ""),
        ("Goals Scored", f"{prof_h['GF']}", f"{prof_a['GF']}", f"{prof_h['GF']-prof_a['GF']:+d}"),
        ("Goals Conceded", f"{prof_h['GA']}", f"{prof_a['GA']}", f"{prof_h['GA']-prof_a['GA']:+d}"),
        ("Goal Difference", f"{prof_h['GD']:+d}", f"{prof_a['GD']:+d}", f"{prof_h['GD']-prof_a['GD']:+d}"),
        ("xG Total", f"{prof_h['xG']:.2f}", f"{prof_a['xG']:.2f}", f"{prof_h['xG']-prof_a['xG']:+.2f}"),
        ("xG / Match", f"{prof_h['xG/m']:.2f}", f"{prof_a['xG/m']:.2f}", f"{prof_h['xG/m']-prof_a['xG/m']:+.2f}"),
        ("xGA / Match", f"{prof_h['xGA/m']:.2f}", f"{prof_a['xGA/m']:.2f}", f"{prof_h['xGA/m']-prof_a['xGA/m']:+.2f}"),
        ("Shots on Target", f"{prof_h['SoT']}", f"{prof_a['SoT']}", f"{prof_h['SoT']-prof_a['SoT']:+d}"),
        ("Avg Possession", f"{prof_h['Poss%']:.0f}%", f"{prof_a['Poss%']:.0f}%", ""),
        ("Clean Sheets", f"{prof_h['CS']}", f"{prof_a['CS']}", ""),
        ("Finishing (G/xG)", f"{prof_h['Finishing']:.2f}", f"{prof_a['Finishing']:.2f}", ""),
    ]
    for label, hv, av, edge_str in rows:
        print(f"  {label:<28} {hv:>12} {av:>12} {edge_str:>12}")


def print_match_log(team_label, matches):
    print(f"\n\n{'─' * 72}")
    print(f"  🗓️  MATCH-BY-MATCH — {team_label.upper()}")
    print(f"{'─' * 72}")
    print(f"  {'Date':<12} {'Stage':<8} {'Opponent':<22} "
          f"{'Score':>6} {'xG':>5} {'xGA':>5} {'Poss':>5}")
    for m in matches:
        print(f"  {m['date']:<12} {m['stage']:<8} {m['opp']:<22} "
              f"{m['gf']}-{m['ga']:>2}  {m['xg']:>5.2f} {m['xga']:>5.2f} "
              f"{m['poss']:>4}%")


def print_bench_impact(home_team, away_team, bench_h, bench_a):
    print(f"\n\n{'─' * 72}")
    print("  🔄 BENCH IMPACT ANALYSIS")
    print(f"{'─' * 72}")
    for label, bench in ((home_team, bench_h), (away_team, bench_a)):
        print(f"\n  {label} key substitutes:")
        for p in bench:
            print(f"    {p['name']:<20} {p['pos']:<3}  G:{p.get('wc_goals',0)} A:{p.get('wc_assists',0)}  "
                  f"Impact: {p['impact']}  — {p['notes']}")
    print(f"\n  Bench impact score: {home_team} {compute_bench_impact(bench_h):.3f}  |  "
          f"{away_team} {compute_bench_impact(bench_a):.3f}")


def print_head_to_head(long, elo_home, elo_away, home_team, away_team, match_date):
    print(f"\n\n{'─' * 72}")
    print("  🤝 HEAD-TO-HEAD HISTORY")
    print(f"{'─' * 72}")
    h2h_matches = long[(long["team"] == elo_home) & (long["opp"] == elo_away)
                       & (long["date"] < pd.Timestamp(match_date))].sort_values("date")
    if len(h2h_matches) == 0:
        print("  No previous meetings found in dataset")
        return
    for _, row in h2h_matches.tail(10).iterrows():
        res = "W" if row["result"] == 1.0 else ("D" if row["result"] == 0.5 else "L")
        print(f"  {str(row['date'].date()):<12}  {home_team} "
              f"{int(row['gf'])}-{int(row['ga'])} {away_team}  ({res})")
    n_h2h, wr_h2h, gd_h2h = h2h_as_of(long, elo_home, elo_away, match_date)
    print(f"\n  Total meetings: {int(n_h2h)}, {home_team} win-rate: {wr_h2h:.2f}, "
          f"avg GD: {gd_h2h:+.1f}")


def print_charts_saved(match_date, chart_paths):
    print(f"\n\n{'─' * 72}")
    print(f"  🖼️  Charts saved to predictions/{match_date}/")
    for c in chart_paths:
        print(f"    -> {c}")
    print(f"{'─' * 72}\n")
