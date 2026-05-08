"""
DataFrame-based z-score scaling utilities.

This module defines a small reusable scaler for tabular feature columns.
It fits per-column means and standard deviations on training data, can
impute missing values with training means, and applies the same transform
to train/validation/test splits. The fitted statistics can also be saved
to and loaded from JSON.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List
import json
import pandas as pd

class ColStats:
    mu: float
    sigma: float

#fit a scaler and store
class ZScaler:
    stats: Dict[str, ColStats]

    def fit(df: pd.DataFrame, cols: Iterable[str], eps: float = 1e-8) -> "ZScaler":
        stats: Dict[str, ColStats] = {}
        for c in cols:
            m = float(df[c].mean())
            s = float(df[c].std(ddof=0))
            if s < eps: s = eps
            stats[c] = ColStats(mu=m, sigma=s)
        return ZScaler(stats=stats)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, st in self.stats.items():
            if c in out.columns:
                out[c] = (out[c] - st.mu) / st.sigma
        return out

    def impute_means(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, st in self.stats.items():
            if c in out.columns:
                out[c] = out[c].fillna(st.mu)
        return out

    def to_json(self, path: str) -> None:
        payload = {c: asdict(st) for c, st in self.stats.items()}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ZScaler":
        with open(path, "r") as f:
            payload = json.load(f)
        stats = {c: ColStats(**st) for c, st in payload.items()}
        return ZScaler(stats=stats)
