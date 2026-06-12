#!/usr/bin/env bash
#
# One-command launcher for the FIFA 2026 prediction system.
#
#   bash run.sh
#
# Works on regular Linux/macOS and on Termux (Android). On Termux it first
# installs the required system packages, then runs the whole pipeline
# (download data → build features → train models → simulate tournament)
# and finally serves the browser UI at http://localhost:8000.
#
# Environment overrides:
#   RUNS=50000 bash run.sh    # number of Monte Carlo runs
#   PORT=8080  bash run.sh    # web UI port
#
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
RUNS="${RUNS:-}"

# ── Detect Termux ─────────────────────────────────────────────────────────────
IS_TERMUX=0
if [ -n "$TERMUX_VERSION" ]; then
  IS_TERMUX=1
elif [ -n "$PREFIX" ]; then
  case "$PREFIX" in
    *com.termux*) IS_TERMUX=1 ;;
  esac
fi

if [ "$IS_TERMUX" = 1 ]; then
  echo "==> Termux detected"
  echo "==> Installing system packages (first run can take several minutes)…"
  pkg update -y >/dev/null 2>&1 || true
  # Build toolchain (needed to pip-install lightgbm) + science stack binaries
  pkg install -y python clang cmake make pkg-config libomp openblas which >/dev/null 2>&1 || true
  # TUR repo provides prebuilt numpy/scipy/pandas/matplotlib — far faster than pip builds
  pkg install -y tur-repo >/dev/null 2>&1 || true
  pkg install -y python-numpy python-scipy python-pandas matplotlib >/dev/null 2>&1 || true
  # Fewer Monte Carlo runs by default on a phone
  [ -z "$RUNS" ] && RUNS=5000
else
  [ -z "$RUNS" ] && RUNS=20000
fi

# ── Install the package ───────────────────────────────────────────────────────
echo "==> Installing wm package and Python dependencies…"
python -m pip install -e . --quiet

# ── Pipeline (idempotent: each step is skipped if its output already exists) ──
if [ ! -f data/raw/results.csv ]; then
  echo "==> [1/4] Downloading match data (~48k international matches)…"
  python -m wm.cli ingest --source results
else
  echo "==> [1/4] Match data already downloaded ✓"
fi

if [ ! -f data/processed/features.parquet ] && [ ! -f data/processed/features.pkl ]; then
  echo "==> [2/4] Building feature matrix (Elo, form, travel, climate)…"
  python -m wm.cli build-features
else
  echo "==> [2/4] Feature matrix already built ✓"
fi

if [ ! -f models/wdl_clf.lgb ]; then
  echo "==> [3/4] Training models (LightGBM + calibration)…"
  python -m wm.cli train
else
  echo "==> [3/4] Models already trained ✓"
fi

if [ ! -f reports/simulation/results.json ]; then
  echo "==> [4/4] Simulating tournament ($RUNS Monte Carlo runs)…"
  python -m wm.cli simulate --runs "$RUNS"
else
  echo "==> [4/4] Simulation results already exist ✓"
fi

# ── Serve the browser UI ──────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  ⚽ FIFA 2026 Predictor ready!"
echo "  Open http://localhost:$PORT in your browser"
echo "=============================================="
echo ""

# On Termux, open the Android browser automatically
if [ "$IS_TERMUX" = 1 ] && command -v termux-open-url >/dev/null 2>&1; then
  (sleep 3 && termux-open-url "http://localhost:$PORT") &
fi

exec python -m wm.cli serve --port "$PORT"
