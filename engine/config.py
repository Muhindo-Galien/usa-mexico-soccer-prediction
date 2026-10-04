"""Shared constants for the prediction engine. Match-specific values
(teams, date, venue, colors) stay in the top-level predict_*.py scripts —
only values that apply across every matchup live here."""

CACHE_DIR   = "data_cache"
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

NAME_MAP = {
    "USA": "United States", "Korea Republic": "South Korea",
    "Republic of Ireland": "Ireland", "Türkiye": "Turkey",
    "Cape Verde": "Cabo Verde", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Congo DR": "DR Congo", "Congo": "Republic of the Congo",
}

TEAM_FEATURES = [
    "neutral", "tournament_weight", "home_elo", "away_elo", "elo_diff",
    "home_win5", "away_win5", "home_gd5", "away_gd5",
    "home_win10", "away_win10", "home_rest_days", "away_rest_days",
    "h2h_n", "h2h_home_winrate", "h2h_home_gd",
]

TRAIN_START    = "2006-01-01"
VAL_START      = "2023-01-01"
ELO_BASE       = 1500.0
ELO_K          = 32
ELO_HOME_BONUS = 60

N_SIMULATIONS      = 10_000
DIXON_COLES_RHO    = -0.13
MAX_GOALS          = 7
TOURNAMENT_AVG_XG  = 1.40

# Later tournament stages count for more when averaging a team's xG/xGA.
STAGE_WEIGHT = {"Group": 1.0, "R32": 1.2, "R16": 1.4, "QF": 1.6, "SF": 1.8}
