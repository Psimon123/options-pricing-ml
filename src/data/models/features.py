"""
features.py  —  shared feature engineering for brute_force.py and theory_informed.py

Extracted here to avoid the identical _bs_delta_proxy() function being
defined twice. Import with:  from features import add_features
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from math import log, sqrt
from scipy.stats import norm


def _delta_proxy(row) -> float:
    """
    BS delta at the expiry's calibrated flat_vol.
    Encodes moneyness and T jointly without using mid price (no leakage).
    For puts, returns 1 - delta so both calls and puts are on [0, 1].
    """
    try:
        S, K, T = float(row["S"]), float(row["K"]), float(row["T"])
        sigma   = float(row["flat_vol"])
        if T <= 0 or sigma <= 0:
            return 0.5
        d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
        delta = float(norm.cdf(d1))
        return delta if row["kind"] == "call" else 1.0 - delta
    except Exception:
        return float(row.get("moneyness", 1.0))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by both RF models."""
    df = df.copy()
    df["moneyness"]     = df["K"] / df["S"]
    df["log_moneyness"] = np.log(df["K"] / df["S"])
    df["delta_proxy"]   = df.apply(_delta_proxy, axis=1)
    df["intrinsic"]     = np.where(
        df["kind"] == "call",
        np.maximum(df["S"] - df["K"], 0.0),
        np.maximum(df["K"] - df["S"], 0.0),
    )
    return df.replace([np.inf, -np.inf], np.nan)