# ⚽ WM — FIFA World Cup 2026 Prediction System

An AI that predicts the FIFA World Cup 2026 (USA/Canada/Mexico, 48 teams).
It trains on ~48,000 international matches since 1872 and combines rolling Elo
ratings, recent form, player squad strength, FIFA rankings, travel distance,
altitude, and venue climate into a calibrated LightGBM ensemble, then runs
Monte Carlo simulations of the entire tournament — exact 2026 format with
12 groups, best-thirds qualification, and the Round-of-32 bracket.

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
wm ingest --source all       # match data + player ratings + FIFA rankings (free, no auth)
wm build-features            # leak-free feature matrix
wm train                     # LightGBM W/D/L + Poisson goals + calibration
wm evaluate                  # log-loss / Brier / RPS vs Elo baseline
wm simulate --runs 100000    # Monte Carlo tournament simulation
wm report                    # static HTML report
wm serve                     # interactive browser UI
wm predict Brazil Germany    # single match prediction
```

All data sources download automatically — match results, FIFA-24 player
attributes, and historical FIFA rankings come from free GitHub mirrors with
no authentication. Optionally, an official EA FC ratings export from
[Kaggle](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)
saved to `data/raw/players/ea_fc_ratings.csv` takes precedence over the
auto-downloaded attribute-based ratings.

## How it works

| Layer | What it does |
|---|---|
| **Data** | International results 1872–present (GitHub), EA FC squad ratings, FIFA rankings, Open-Meteo climate normals for all 18 venues |
| **Features** | Rolling Elo (K scaled by tournament tier), form windows (5/10/15), rest days, head-to-head, travel km, altitude, heat/humidity, confederation strength |
| **Models** | LightGBM multiclass W/D/L + two LightGBM Poisson goal regressors, blended and temperature-calibrated; Dixon-Coles as reference |
| **Split** | 80% of time-sorted matches → training, last 20% → validation (no temporal leakage) |
| **Simulation** | Exact 2026 format: 12 groups → top 2 + 8 best thirds → R32 → … → Final, with extra time, penalties, and per-run squad-strength noise |

## Development

```bash
pip install -e ".[dev]"
pytest
```
