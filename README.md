# ⚽ WM — FIFA World Cup 2026 Prediction System

An AI that predicts the FIFA World Cup 2026 (USA/Canada/Mexico, 48 teams).
It trains on ~48,000 international matches since 1872 and combines rolling Elo
ratings, recent form, real club-football performance (goals, xG, fouls, cards
from the top-5 European leagues), FIFA rankings, travel distance, altitude,
and venue climate into a calibrated LightGBM + neural-net ensemble, then runs
Monte Carlo simulations of the entire tournament — official 2026 bracket with
12 groups, best-thirds qualification, and the real Round-of-32 venues.

## Quick start (one command)

```bash
bash run.sh
```

That downloads the data, builds features, trains the models, simulates the
tournament, and opens the browser UI at <http://localhost:8000>.

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukas04545/wm/blob/main/colab/WM_FIFA_2026.ipynb)

Open [`colab/WM_FIFA_2026.ipynb`](colab/WM_FIFA_2026.ipynb) in Colab and
`Runtime → Run all`. It runs the whole pipeline, shows the championship-odds
chart and table inline, and can serve the interactive dashboard through
Colab's port proxy (a clickable link is printed).

### Termux (Android)

```bash
pkg install -y git && git clone https://github.com/lukas04545/wm.git && cd wm && bash run.sh
```

`run.sh` auto-detects Termux: it installs the required system packages
(prebuilt numpy/scipy/pandas from the TUR repo, clang/cmake for LightGBM),
uses a phone-friendly number of simulation runs, and opens the Android
browser automatically when the UI is ready.

Options: `RUNS=50000 bash run.sh` (more Monte Carlo runs), `PORT=8080 bash run.sh`.

## Manual pipeline

```bash
pip install -e .
wm ingest --source all       # match results + FBref club stats + FIFA rankings (free, no auth)
wm build-features            # leak-free feature matrix
wm train                     # LightGBM W/D/L + Poisson goals + calibration
wm evaluate                  # log-loss / Brier / RPS vs Elo baseline
wm simulate --runs 100000    # Monte Carlo tournament simulation
wm report                    # static HTML report
wm serve                     # interactive browser UI
wm predict Brazil Germany    # single match prediction
```

All data sources download automatically with no authentication — match
results, **real FBref top-5-league club stats** (goals, assists, xG, minutes,
plus discipline: fouls committed/drawn, cards, tackles, interceptions,
aerials), and historical FIFA rankings, all from free GitHub mirrors. The
club stats are aggregated to each nation per season and joined to
internationals leak-free (most recent completed season only). No EA Sports /
FIFA video-game ratings are used — every player number is real on-pitch
performance.

## How it works

| Layer | What it does |
|---|---|
| **Data** | International results 1872–present (GitHub), real FBref top-5-league club stats (goals, assists, xG, fouls, cards, tackles, aerials, league strength), FIFA rankings, Open-Meteo climate normals for all 18 venues |
| **Features** | Rolling Elo + attack/defence goal ratings, opponent-adjusted form (performance vs Elo expectation, schedule strength), EWMA goals, form windows (5/10/15), rest days, head-to-head, real club-form per nation (goals/xG/fouls/cards/tackles per 90, league-strength talent score), travel km, altitude, heat/humidity, confederation strength, FIFA rank, GDP per capita, population, World Cup pedigree, host-nation status (131 features) |
| **Models** | Three-branch ensemble: LightGBM W/D/L classifier + a bagged backprop neural net (NumPy MLP) + LightGBM Poisson goal regressors with Dixon-Coles low-score correction. Stacked meta-learner blend + vector-scaling calibration, both selected on validation |
| **Split** | 80% of time-sorted matches → training, last 20% → validation (no temporal leakage) |
| **Simulation** | Official 2026 bracket: 12 groups → top 2 + 8 best thirds → R32 (FIFA matches 73–88 with real venues) → … → Final at MetLife. Third-place slots filled by constraint-respecting matching; scoreline grids reshaped to the calibrated ensemble W/D/L, with extra time, penalties, and per-run squad-strength noise |
| **Live mode** | `wm simulate` (default) reads the real 72-fixture schedule from the dataset: real venues per match (Azteca altitude, Houston heat…), real host/neutral flags, and **already-played 2026 results are locked in** instead of re-simulated. Re-run `wm ingest --source results --force && wm build-features && wm simulate` during the tournament to update the odds after every matchday |

## Development

```bash
pip install -e ".[dev]"
pytest
```
