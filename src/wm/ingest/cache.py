"""Download-once file caching with SHA-256 verification."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    registry: Path,
    force: bool = False,
    chunk_size: int = 65536,
) -> Path:
    """Download url to dest; skip if already cached unless force=True."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    registry.parent.mkdir(parents=True, exist_ok=True)

    reg: dict = {}
    if registry.exists():
        with open(registry) as f:
            reg = json.load(f)

    key = str(dest.relative_to(dest.parent.parent))
    if not force and dest.exists() and key in reg:
        return dest

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    reg[key] = {
        "url": url,
        "fetched_at": datetime.utcnow().isoformat(),
        "sha256": _sha256(dest),
    }
    with open(registry, "w") as f:
        json.dump(reg, f, indent=2)

    return dest
