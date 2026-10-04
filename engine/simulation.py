"""Monte Carlo goal simulation, with lambdas blended from player-level
squad strength and tournament xG."""

import numpy as np

from .config import TOURNAMENT_AVG_XG, N_SIMULATIONS
from .players import compute_bench_impact


def compute_player_derived_lambdas(squad_h, squad_a, prof_h, prof_a,
                                    bench_h, bench_a, matches_h, matches_a,
                                    knockout_regression=0.95,
                                    tournament_avg_xg=TOURNAMENT_AVG_XG):
    """
    Derive expected goals (lambda) for each team using a blend of:
      - Player-level attacking/defensive strength indices
      - Tournament xG data (stage-weighted)
      - Opponent quality adjustment

    This is the KEY improvement over team-level models: the lambda
    parameters are informed by individual player quality, not just
    team-level averages.
    """
    # xG-based baseline
    h_att = prof_h["xG/m_w"] / tournament_avg_xg
    h_def = prof_h["xGA/m_w"] / tournament_avg_xg
    a_att = prof_a["xG/m_w"] / tournament_avg_xg
    a_def = prof_a["xGA/m_w"] / tournament_avg_xg

    # Blend player-level and xG-level
    player_weight = 0.40
    xg_weight = 0.60

    # Home team expected goals
    lambda_h_player = tournament_avg_xg * (squad_h["attack"] / squad_a["defense"]) * 1.1
    lambda_h_xg = tournament_avg_xg * h_att * a_def * knockout_regression
    lambda_h = player_weight * lambda_h_player + xg_weight * lambda_h_xg

    # Away team expected goals
    lambda_a_player = tournament_avg_xg * (squad_a["attack"] / squad_h["defense"]) * 1.1
    lambda_a_xg = tournament_avg_xg * a_att * h_def * knockout_regression
    lambda_a = player_weight * lambda_a_player + xg_weight * lambda_a_xg

    # Bench depth adjustment (potential sub impact)
    lambda_h *= (1.0 + 0.03 * compute_bench_impact(bench_h))
    lambda_a *= (1.0 + 0.03 * compute_bench_impact(bench_a))

    # Computed momentum: normalized GD trend over last 3 knockout matches
    # Applied symmetrically to both teams using the same formula
    def _knockout_momentum(matches):
        ko = [m for m in matches if m["stage"] in ("R32", "R16", "QF", "SF")][-3:]
        if not ko:
            return 0.0
        gd_trend = sum((m["gf"] - m["ga"]) for m in ko) / len(ko)
        # Sigmoid-like mapping: large positive GD → small boost, negative → drag
        return 0.01 * np.tanh(gd_trend / 2.0)

    lambda_h *= (1.0 + _knockout_momentum(matches_h))
    lambda_a *= (1.0 + _knockout_momentum(matches_a))

    return lambda_h, lambda_a


def simulate_match(lambda_h, lambda_a, n_sims=N_SIMULATIONS):
    """
    Monte Carlo simulation of the match using Poisson-distributed goals.
    Returns win/draw/loss probabilities and score distribution.
    """
    goals_h = np.random.poisson(lambda_h, n_sims)
    goals_a = np.random.poisson(lambda_a, n_sims)

    home_wins = np.sum(goals_h > goals_a)
    draws = np.sum(goals_h == goals_a)
    away_wins = np.sum(goals_h < goals_a)

    p_home = home_wins / n_sims
    p_draw = draws / n_sims
    p_away = away_wins / n_sims

    # Score distribution
    scores = {}
    for h, a in zip(goals_h, goals_a):
        key = (int(h), int(a))
        scores[key] = scores.get(key, 0) + 1
    for key in scores:
        scores[key] /= n_sims

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "avg_goals_home": np.mean(goals_h), "avg_goals_away": np.mean(goals_a),
        "scores": scores,
    }
