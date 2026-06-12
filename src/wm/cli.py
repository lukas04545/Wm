"""Command-line interface for the WM prediction pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="wm", help="FIFA World Cup 2026 prediction system")
console = Console()


@app.command()
def ingest(
    source: str = typer.Option("all", help="Data source: results|players|rankings|weather|all"),
    force: bool = typer.Option(False, help="Re-download even if cached"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
):
    """Download and cache raw data from public sources."""
    from wm import config as cfg_mod
    from wm.ingest import results as res_mod
    from wm.ingest import player_ratings as pr_mod
    from wm.ingest import fifa_rankings as rank_mod

    cfg = cfg_mod.load(config)
    raw_dir = cfg.path("data_raw")
    registry = raw_dir / "registry.json"

    sources = ["results", "players", "rankings", "weather"] if source == "all" else [source]

    for src in sources:
        console.print(f"[bold blue]Ingesting:[/bold blue] {src}")
        if src == "results":
            res_mod.fetch(raw_dir, registry, force=force)
            console.print(f"  ✓ international results → {raw_dir}/results.csv")
        elif src == "players":
            try:
                dest = pr_mod.fetch(raw_dir, registry, force=force)
                console.print(f"  ✓ FIFA-24 player attributes → {dest}")
                console.print(
                    "    (optional upgrade: Kaggle EA FC dataset with official overalls\n"
                    f"     → save as {raw_dir}/players/ea_fc_ratings.csv)"
                )
            except Exception as e:
                console.print(f"  ⚠ Player ratings download failed ({e}) — feature is optional")
        elif src == "rankings":
            try:
                dest = rank_mod.fetch(raw_dir, registry, force=force)
                console.print(f"  ✓ FIFA rankings 1992–2024 → {dest}")
            except Exception as e:
                console.print(f"  ⚠ FIFA rankings download failed ({e}) — feature is optional")
        elif src == "weather":
            console.print("  ✓ Weather: fetched on-demand during build-features")

    console.print("[green]✓ Ingest complete[/green]")


@app.command(name="build-features")
def build_features(
    start: str = typer.Option("1990-01-01", help="Start date for training data"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
):
    """Build feature matrix from raw data."""
    import pandas as pd
    from wm import config as cfg_mod
    from wm.data import build as build_mod
    from wm.ingest import results as res_mod
    from wm.ingest import player_ratings as pr_mod
    from wm.ingest import fifa_rankings as rank_mod
    from wm.features.matrix import build_matrix

    cfg = cfg_mod.load(config)
    raw_dir = cfg.path("data_raw")
    processed_dir = cfg.path("data_processed")
    interim_dir = cfg.path("data_interim")

    console.print("[bold blue]Loading results...[/bold blue]")
    results_df, shootouts_df = res_mod.load(raw_dir)
    console.print(f"  ✓ {len(results_df):,} matches loaded")

    matches = build_mod.build(results_df)
    matches = matches[matches["date"] >= pd.Timestamp(start)]
    console.print(f"  ✓ {len(matches):,} matches after {start}")

    # Save canonical matches
    build_mod.save(matches, interim_dir / "matches.parquet")

    # Load optional enrichment data
    squad_df = pr_mod.get_squad_features(raw_dir)
    if squad_df is not None:
        console.print(f"  ✓ Squad ratings: {len(squad_df)} teams")
    else:
        console.print("  ⚠ Player ratings not found (optional) - model will use Elo/form only")

    rankings_df = rank_mod.load(raw_dir)
    if rankings_df is not None:
        console.print(f"  ✓ FIFA rankings: {len(rankings_df):,} entries")
    else:
        console.print("  ⚠ FIFA rankings not found (optional)")

    console.print("[bold blue]Building feature matrix...[/bold blue]")
    feat_df = build_matrix(matches, cfg, squad_df=squad_df, rankings_df=rankings_df)
    console.print(f"  ✓ Feature matrix: {len(feat_df)} rows × {len(feat_df.columns)} features")

    from wm.data.io import save_df
    feat_path = save_df(feat_df, processed_dir / "features.parquet")
    console.print(f"[green]✓ Features saved to {feat_path}[/green]")


@app.command()
def train(
    model: str = typer.Option("all", help="Models to train: all | all+dc (adds slow Dixon-Coles reference fit)"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
):
    """Train the prediction models (W/D/L classifier + goal regressors + calibration)."""
    import pandas as pd
    from wm import config as cfg_mod
    from wm.eval.splits import split
    from wm.models import persistence
    from wm.models.train import train_ensemble

    cfg = cfg_mod.load(config)
    processed_dir = cfg.path("data_processed")
    models_dir = cfg.path("models")

    console.print("[bold blue]Loading feature matrix...[/bold blue]")
    from wm.data.io import load_df
    feat_df = load_df(processed_dir / "features.parquet")
    train_df, val_df, test_df = split(feat_df, cfg.splits)
    console.print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}")

    predictor, info = train_ensemble(
        train_df, val_df, cfg,
        progress=lambda m: console.print(f"[bold blue]{m}...[/bold blue]"),
    )
    bw = info["blend_weights"]
    console.print(
        f"  ✓ blend weights: GBM {bw[0]:.3f} | NN {bw[1]:.3f} | Poisson {bw[2]:.3f} "
        f"| NN val log-loss {info['nn_val_loss']:.4f}"
    )
    console.print(
        f"  ✓ blend: {'stacked meta-learner' if info.get('use_stacker') else 'convex weights'} "
        f"| calibration: {info.get('calibrator', '?')} | DC rho: {info.get('dc_rho', 0):.4f}"
    )

    dc = None
    if model == "all+dc":
        # Reference model only — MLE over hundreds of team parameters is slow
        console.print("[bold blue]Fitting Dixon-Coles model (slow)...[/bold blue]")
        from wm.models.goals_poisson import DixonColes
        from wm.data import build as build_mod
        interim_dir = cfg.path("data_interim")
        matches = build_mod.load(interim_dir / "matches.parquet")
        dc = DixonColes(xi=0.0018)
        dc.fit(matches[matches["date"] >= pd.Timestamp("2018-01-01")])
        console.print("  ✓ Dixon-Coles fitted")

    meta = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    persistence.save(predictor, dc, models_dir, meta=meta)
    console.print(f"[green]✓ Models saved to {models_dir}[/green]")


@app.command()
def evaluate(
    split_name: str = typer.Option("test", help="Split to evaluate: val|test"),
    baselines: bool = typer.Option(True, help="Also evaluate baseline models"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
):
    """Evaluate model on held-out data and print metrics."""
    import pandas as pd
    import numpy as np
    from wm import config as cfg_mod
    from wm.eval.splits import split
    from wm.eval.backtest import run_backtest
    from wm.models import persistence, wdl_classifier, baselines as bl
    from wm.features.matrix import TARGET_WDL

    cfg = cfg_mod.load(config)
    from wm.data.io import load_df
    feat_df = load_df(cfg.path("data_processed") / "features.parquet")
    train_df, val_df, test_df = split(feat_df, cfg.splits)
    eval_df = val_df if split_name == "val" else test_df

    predictor, dc = persistence.load(cfg.path("models"))
    result = predictor.predict(eval_df)
    probs = np.column_stack([result["p_away_win"], result["p_draw"], result["p_home_win"]])

    run_backtest(eval_df, probs, cfg.path("reports"), name=f"ensemble_{split_name}")

    if baselines:
        elo_baseline = bl.EloBaseline().fit(train_df)
        draw_baseline = bl.DrawRateBaseline().fit(train_df)
        run_backtest(eval_df, elo_baseline.predict_proba(eval_df), cfg.path("reports"), name="elo_baseline")
        run_backtest(eval_df, draw_baseline.predict_proba(eval_df), cfg.path("reports"), name="draw_baseline")


@app.command()
def backtest(
    folds: int = typer.Option(4, help="Number of rolling-origin time folds"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
):
    """
    Rolling-origin backtest: retrain on an expanding window and evaluate on the
    next time block, repeated across folds. Confirms accuracy generalizes
    rather than being tuned to one validation split.
    """
    import numpy as np
    import pandas as pd
    from wm import config as cfg_mod
    from wm.data.io import load_df
    from wm.models.train import train_ensemble
    from wm.models import baselines as bl
    from wm.eval.metrics import evaluate as eval_metrics
    from wm.features.matrix import TARGET_WDL

    cfg = cfg_mod.load(config)
    feat_df = load_df(cfg.path("data_processed") / "features.parquet").sort_values("date").reset_index(drop=True)
    n = len(feat_df)

    # Expanding-window folds over the most recent half of the data
    start = n // 2
    bounds = np.linspace(start, n, folds + 1).astype(int)

    rows = []
    for k in range(folds):
        tr_end, te_end = bounds[k], bounds[k + 1]
        train_df = feat_df.iloc[:tr_end]
        test_df = feat_df.iloc[tr_end:te_end]
        # inner validation = last 15% of the training window (for blend/calibration)
        cut = int(len(train_df) * 0.85)
        inner_tr, inner_val = train_df.iloc[:cut], train_df.iloc[cut:]

        console.print(f"[bold blue]Fold {k+1}/{folds}[/bold blue] "
                      f"train≤{train_df['date'].iloc[-1].date()} "
                      f"test n={len(test_df)}")
        predictor, _ = train_ensemble(inner_tr, inner_val, cfg)

        out = predictor.predict(test_df)
        probs = np.column_stack([out["p_away_win"], out["p_draw"], out["p_home_win"]])
        y = test_df[TARGET_WDL].astype(int).values
        m = eval_metrics(y, probs)

        elo = bl.EloBaseline().fit(inner_tr)
        m_elo = eval_metrics(y, elo.predict_proba(test_df))

        rows.append({"fold": k + 1, "n": len(test_df),
                     "ensemble_ll": m["log_loss"], "elo_ll": m_elo["log_loss"],
                     "ensemble_rps": m["rps"]})
        console.print(f"  ensemble log-loss {m['log_loss']:.4f}  |  Elo {m_elo['log_loss']:.4f}")

    res = pd.DataFrame(rows)
    table = Table(title=f"Rolling backtest ({folds} folds)", show_header=True)
    for col in ["Fold", "n", "Ensemble LL", "Elo LL", "Ensemble RPS"]:
        table.add_column(col, justify="right")
    for _, r in res.iterrows():
        table.add_row(str(int(r["fold"])), str(int(r["n"])),
                      f"{r['ensemble_ll']:.4f}", f"{r['elo_ll']:.4f}", f"{r['ensemble_rps']:.4f}")
    console.print(table)

    w = res["n"] / res["n"].sum()
    ens = float((res["ensemble_ll"] * w).sum())
    elo = float((res["elo_ll"] * w).sum())
    console.print(f"[bold green]Weighted mean log-loss[/bold green]  "
                  f"ensemble {ens:.4f}  vs  Elo {elo:.4f}  "
                  f"(Δ {elo - ens:+.4f})")


@app.command()
def simulate(
    runs: int = typer.Option(100_000, help="Number of Monte Carlo runs"),
    seed: int = typer.Option(42),
    out: Path = typer.Option("reports/simulation/results.json", help="Output JSON path"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
):
    """Run Monte Carlo tournament simulation."""
    import pandas as pd
    from wm import config as cfg_mod
    from wm.models import persistence
    from wm.ingest import player_ratings as pr_mod, fifa_rankings as rank_mod
    from wm.features.elo_rolling import get_current_ratings
    from wm.sim.monte_carlo import TournamentSimulator
    from wm import config as cfg_mod2

    cfg = cfg_mod.load(config)
    cfg.simulation.n_runs = runs
    cfg.simulation.seed = seed

    wc_cfg = cfg_mod2.load_wc2026()
    groups = {grp: data["teams"] for grp, data in wc_cfg["groups"].items()}

    console.print(f"[bold blue]Loading models...[/bold blue]")
    predictor, dc = persistence.load(cfg.path("models"))

    # Compute current team state (Elo, att/def ratings, form) from history
    from wm.data import build as build_mod
    from wm.features.state import compute_team_state
    interim_dir = cfg.path("data_interim")
    matches = build_mod.load(interim_dir / "matches.parquet")
    team_state = compute_team_state(matches, cfg)
    elo_ratings = {t: s["elo"] for t, s in team_state.items()}
    console.print(f"  ✓ Team state (Elo, att/def, form) for {len(team_state)} teams")

    squad_df = pr_mod.get_squad_features(cfg.path("data_raw"))
    rankings_df = rank_mod.load(cfg.path("data_raw"))

    console.print(f"[bold blue]Simulating {runs:,} tournaments...[/bold blue]")
    simulator = TournamentSimulator(
        groups=groups,
        predictor=predictor,
        elo_ratings=elo_ratings,
        cfg=cfg,
        squad_df=squad_df,
        rankings_df=rankings_df,
        team_state=team_state,
    )
    results = simulator.run(runs, seed=seed)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"[green]✓ Results saved to {out}[/green]")

    # Print top 10 favorites
    table = Table(title="Top 10 Championship Favorites", show_header=True)
    for col in ["Team", "Champion %", "Final %", "SF %", "QF %"]:
        table.add_column(col)
    sorted_teams = sorted(
        [(t, d) for t, d in results.items() if not str(t).startswith("_")],
        key=lambda x: x[1].get("champion", 0),
        reverse=True,
    )[:10]
    for team, data in sorted_teams:
        table.add_row(
            team,
            f"{data.get('champion',0)*100:.1f}%",
            f"{data.get('final',0)*100:.1f}%",
            f"{data.get('semifinal',0)*100:.1f}%",
            f"{data.get('quarterfinal',0)*100:.1f}%",
        )
    console.print(table)


@app.command()
def report(
    sim_results: Path = typer.Option("reports/simulation/results.json"),
    out: Path = typer.Option("reports/wc2026_report.html"),
    config: Path = typer.Option("config/default.yaml"),
):
    """Generate HTML prediction report."""
    from wm.report.render import render_html

    with open(sim_results) as f:
        results = json.load(f)

    render_html(results, metrics=None, out_path=out)


@app.command()
def predict(
    team_a: str = typer.Argument(..., help="Home team name"),
    team_b: str = typer.Argument(..., help="Away team name"),
    venue: str = typer.Option("MetLife Stadium", help="Venue name"),
    date: str = typer.Option("2026-07-01", help="Match date YYYY-MM-DD"),
    config: Path = typer.Option("config/default.yaml"),
):
    """Predict the outcome of a single match."""
    import pandas as pd
    from wm import config as cfg_mod
    from wm.models import persistence
    from wm.features.state import compute_team_state
    from wm.sim.match_sampler import build_fixture_row
    from wm.data import build as build_mod
    from wm.ingest import player_ratings as pr_mod, fifa_rankings as rank_mod

    cfg = cfg_mod.load(config)
    predictor, _ = persistence.load(cfg.path("models"))

    matches = build_mod.load(cfg.path("data_interim") / "matches.parquet")
    team_state = compute_team_state(matches, cfg)
    elo = {t: s["elo"] for t, s in team_state.items()}

    squad_df = pr_mod.get_squad_features(cfg.path("data_raw"))
    rankings_df = rank_mod.load(cfg.path("data_raw"))

    row = build_fixture_row(
        home=team_a, away=team_b,
        elo_home=elo.get(team_a, 1500.0),
        elo_away=elo.get(team_b, 1500.0),
        venue=venue,
        date=pd.Timestamp(date),
        is_neutral=True,
        is_host_home=False,
        squad_df=squad_df,
        rankings_df=rankings_df,
        cfg=cfg,
        team_state=team_state,
    )

    result = predictor.predict(row)
    ph = float(result["p_home_win"][0])
    pd_ = float(result["p_draw"][0])
    pa = float(result["p_away_win"][0])
    lh = float(result["lambda_home"][0])
    la = float(result["lambda_away"][0])

    table = Table(title=f"{team_a} vs {team_b}", show_header=True)
    table.add_column("Outcome")
    table.add_column("Probability")
    table.add_column("Expected goals")
    table.add_row(f"{team_a} win", f"{ph*100:.1f}%", f"{lh:.2f}")
    table.add_row("Draw", f"{pd_*100:.1f}%", "—")
    table.add_row(f"{team_b} win", f"{pa*100:.1f}%", f"{la:.2f}")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Port number"),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev mode)"),
    config: Path = typer.Option("config/default.yaml", help="Config file"),
):
    """Start the web UI server at http://localhost:PORT"""
    import uvicorn
    from wm.web.app import create_app

    console.print(f"[bold green]⚽ FIFA 2026 Predictor[/bold green]")
    console.print(f"   Open [bold blue]http://localhost:{port}[/bold blue] in your browser\n")

    if reload:
        uvicorn.run("wm.web.app:create_app", host=host, port=port, reload=True, factory=True)
    else:
        web_app = create_app(config)
        uvicorn.run(web_app, host=host, port=port)


if __name__ == "__main__":
    app()
