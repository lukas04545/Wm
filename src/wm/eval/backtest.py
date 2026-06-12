"""Backtest model on WC 2018, WC 2022, Euro 2024 tournaments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from wm.eval.metrics import evaluate, calibration_plot
from wm.features.matrix import TARGET_WDL

console = Console()

TOURNAMENT_FILTERS = {
    "WC 2018": lambda df: df["date"].between("2018-06-14", "2018-07-15") & df.get("is_wc", False),
    "WC 2022": lambda df: df["date"].between("2022-11-20", "2022-12-18") & df.get("is_wc", False),
}


def run_backtest(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    reports_dir: Path,
    name: str = "model",
) -> pd.DataFrame:
    """Evaluate predictions on test set, save calibration plots, print table."""
    y = test_df[TARGET_WDL].astype(int).values
    metrics = evaluate(y, probs, name=name)

    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    result = pd.DataFrame(rows)

    calibration_plot(
        y, probs,
        labels=["Away win", "Draw", "Home win"],
        out_path=reports_dir / f"calibration_{name}.png",
    )

    table = Table(title=f"Backtest: {name}", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for _, r in result.iterrows():
        table.add_row(r["metric"], f"{r['value']:.4f}")
    console.print(table)

    return result
