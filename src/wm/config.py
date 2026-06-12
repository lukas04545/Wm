"""Configuration loading and validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).parent.parent.parent


@dataclass
class EloConfig:
    k_wc: float = 60.0
    k_continental: float = 50.0
    k_qualifier: float = 40.0
    k_friendly: float = 20.0
    home_advantage: float = 100.0
    initial: float = 1500.0


@dataclass
class FeaturesConfig:
    form_windows: list[int] = field(default_factory=lambda: [5, 10, 15])
    h2h_window: int = 10
    min_train_date: str = "1990-01-01"
    weight_halflife_years: float = 8.0


@dataclass
class ModelConfig:
    wdl: dict[str, Any] = field(default_factory=dict)
    goals: dict[str, Any] = field(default_factory=dict)
    blend_weight: float = 0.5


@dataclass
class SimConfig:
    n_runs: int = 100_000
    seed: int = 42
    squad_noise_std: float = 0.03
    max_goals_grid: int = 10
    et_lambda_factor: float = 0.333
    base_penalty_conversion: float = 0.75


@dataclass
class SplitsConfig:
    train_ratio: float = 0.8   # fraction of time-sorted matches used for training
    # Legacy date fields kept for split_by_date() helper
    train_end: str = "2017-12-31"
    val_end: str = "2021-12-31"


@dataclass
class Config:
    paths: dict[str, str] = field(default_factory=dict)
    elo: EloConfig = field(default_factory=EloConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    simulation: SimConfig = field(default_factory=SimConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    root: Path = field(default_factory=lambda: ROOT)

    def path(self, key: str) -> Path:
        p = Path(self.paths.get(key, key))
        if not p.is_absolute():
            p = self.root / p
        p.mkdir(parents=True, exist_ok=True)
        return p


_cfg: Config | None = None


def load(path: Path | None = None) -> Config:
    global _cfg
    if path is None:
        path = ROOT / "config" / "default.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Config()
    cfg.paths = raw.get("paths", {})
    e = raw.get("elo", {})
    cfg.elo = EloConfig(**{k: v for k, v in e.items() if k in EloConfig.__dataclass_fields__})
    ft = raw.get("features", {})
    cfg.features = FeaturesConfig(**{k: v for k, v in ft.items() if k in FeaturesConfig.__dataclass_fields__})
    m = raw.get("model", {})
    cfg.model = ModelConfig(
        wdl=m.get("wdl", {}),
        goals=m.get("goals", {}),
        blend_weight=m.get("blend_weight", 0.5),
    )
    s = raw.get("simulation", {})
    cfg.simulation = SimConfig(**{k: v for k, v in s.items() if k in SimConfig.__dataclass_fields__})
    sp = raw.get("splits", {})
    cfg.splits = SplitsConfig(**{k: v for k, v in sp.items() if k in SplitsConfig.__dataclass_fields__})
    _cfg = cfg
    return cfg


def get() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load()
    return _cfg


def load_wc2026(path: Path | None = None) -> dict:
    if path is None:
        path = ROOT / "config" / "wc2026.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
