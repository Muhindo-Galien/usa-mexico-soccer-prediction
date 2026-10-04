# 2026 USA vs Mexico predictor

A small, readable model that predicts the outcome of USA vs Mexico — the International Friendly happening today, **3 October 2026**, at State Farm Stadium in **Glendale, AZ** — win / draw / loss probabilities, plus a chart you can actually post.

I built this as a "first real ML project" you can fork and learn from: a reusable `engine/` package plus `data/*.csv` files instead of one giant script — see [Project Layout](#project-layout) below.

```
python predict_usa_mexico.py
```

```
============================================================
  USA vs Mexico
  2026-10-03  ·  International Friendly  ·  State Farm Stadium, Glendale, AZ
============================================================
  USA                    win    23.2%
  Draw                          25.5%
  Mexico                 win    51.4%
------------------------------------------------------------
  PICK: Mexico  (51.4%)   [LEAN]
============================================================
```

## How it works

There's no magic here — most of the work is in the features, not the model. Four component models each produce their own win/draw/loss probabilities, then get blended into one ensemble:

- **Player Performance Index.** Each team's actual starting XI (not the full ~26-man squad) gets a position-aware composite score from club form + tournament stats — goals/xG for attackers, tackles/interceptions for defenders, save% for the keeper, etc. These roll up into attack/midfield/defense/GK squad strengths.
- **Monte Carlo simulation.** 10,000 Poisson-simulated matches, with each team's expected goals (λ) derived from the PPI squad strengths blended with their actual 2026 World Cup xG.
- **Dixon-Coles Poisson.** The same λs, run through an adjusted Poisson scoreline matrix (with a low-score correlation correction) to get a full score-probability grid, not just a 3-way split.
- **Team-level XGBoost.** Elo ratings (computed from scratch over every international result since 2006), 5/10-match form, rest days, and head-to-head — the same kind of features a simpler team-vs-team model would use, kept here as an anchor/prior.

The blend weights between those four aren't hand-picked — they're fit by walk-forward backtesting against 4,894 major-tournament matches since 2010, minimizing log-loss. That backtest lands around **64% accuracy** with a log-loss of **0.80** — a real edge, but not a crystal ball (more on that below).

The match itself — teams, date, venue — is configured at the top of `predict_usa_mexico.py`. Player rosters and each team's run through the World Cup come from `data/players/*.csv` and `data/matches/*.csv` (see [Project Layout](#project-layout)).

## Setup

```
pip install -r requirements.txt
python predict_usa_mexico.py
```

No arguments needed — this script is built for one match. Each run prints the full breakdown (squad strength, model comparison, key matchups, head-to-head history) and drops 5 charts plus a `METHODOLOGY.md` writeup in `predictions/2026-10-03/`.

Want a different matchup instead? `predict_today.py` is a separate, general-purpose script that looks up any officially scheduled 2026 World Cup group fixture from `data_cache/fixtures.csv` — e.g. `python predict_today.py "Spain" "Cabo Verde"`. It won't work for this one, since USA vs Mexico isn't on that official schedule.

## Project Layout

```
Soccer_predictions/
├── data/
│   ├── players/usa.csv, mexico.csv        ← full squads, role=starter/bench
│   ├── matches/usa.csv, mexico.csv        ← each team's tournament results
│   └── wc2026_results.csv                 ← shared bracket data (any matchup can reuse)
├── engine/                                 ← reusable modeling logic, zero hardcoded teams
│   ├── config.py        constants (Elo, XGBoost, Poisson params)
│   ├── historical.py    fetch/patch the real results.csv dataset
│   ├── elo_form.py       Elo ratings, rolling form, head-to-head
│   ├── team_model.py      XGBoost team-level classifier
│   ├── players.py          CSV loader + Player Performance Index model
│   ├── tournament.py        CSV loader + tournament profile stats
│   ├── poisson_model.py      Dixon-Coles scoreline matrix
│   ├── simulation.py          Monte Carlo + player-derived lambdas
│   ├── ensemble.py             backtested blend-weight fitting
│   ├── charts.py                all 5 matplotlib chart builders
│   ├── report.py                 generic printed report sections
│   └── methodology.py             methodology/limitations writeup
├── predict_usa_mexico.py                  ← thin: match config + narrative + orchestration
└── predict_today.py                        ← untouched, separate simple script
```

To add or tweak a player, edit the CSV — no Python required. A future matchup script can reuse everything in `engine/` and the shared bracket CSV, and would only need its own `data/players/*.csv`, `data/matches/*.csv`, plus a short script like `predict_usa_mexico.py`.

## What it doesn't do (yet)

Being honest about this matters more than the accuracy number:

- **No live injury/suspension feed.** The starting XIs in `data/players/*.csv` are hand-picked by caps/role, not pulled from an actual pre-match lineup — if someone gets a late knock, the CSV doesn't know.
- **No manager/tactics context.** Formation, press intensity, and "this team only needs a draw" game-state thinking aren't modeled.
- **Friendlies see squad rotation.** With nothing on the line, coaches often rest starters — the model has no "rotation risk" feature for that.
- **PPI weights are hand-tuned, not fitted.** The position-group sub-weights (how much a defender's tackles vs. goals matter) are heuristics, not optimized against outcomes — see `predictions/2026-10-03/METHODOLOGY.md` after a run for the full list.

It also under-calls draws, like most win/draw/loss models do, because draws don't have a clean statistical fingerprint. Soccer is low-scoring and high-variance, so even a good model gets plenty wrong — judge it over many games with log-loss, not on any single result.

## Data

Historical results come from the open [martj42/international_results](https://github.com/martj42/international_results) dataset. Fixtures are the official 2026 schedule.

`data/wc2026_results.csv` and the player stats under `data/players/` are a hand-authored, illustrative fill-in for the 2026 tournament — not a live feed of real results.
## Follow me 
- Tiktok: @galien_codes, 
- Instagram: @galien_codes, 
- Youtube: @galien_codes, 
- Linkedin :  OLAME GALIEN MUHINDO

## License
MIT  Build  with it, use it , learn from it. 
