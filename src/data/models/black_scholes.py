"""
black_scholes.py  —  Merton (1973) Black-Scholes with continuous dividend yield

Three entry points:
  1. bs_price / bs_delta / implied_vol  — core pricing functions
  2. run_on_dataset()                   — builds data/processed/bs_output.csv
  3. flat_vol_predict(train, test)      — BS baseline for compare_models.py

Dividend yield q=0.013 reflects SPY's ~1.3% annual distribution yield.
Flat vol is calibrated per-expiry from ATM implied vols (0.95 ≤ K/S ≤ 1.05).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

Q_SPY: float = 0.013   # SPY continuous dividend yield


# ── Core pricing ──────────────────────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             kind: str = "call", q: float = Q_SPY) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if kind == "call":
        return float(S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
    return float(K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
             kind: str = "call", q: float = Q_SPY) -> float:
    if T <= 0 or sigma <= 0:
        if kind == "call":
            return 1.0 if S > K else (0.0 if S < K else 0.5)
        return -1.0 if S < K else (0.0 if S > K else -0.5)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return float(exp(-q * T) * norm.cdf(d1)) if kind == "call" \
        else float(exp(-q * T) * (norm.cdf(d1) - 1.0))


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                kind: str = "call", q: float = Q_SPY) -> float:
    """Returns np.nan if no solution exists in [1e-6, 5.0]."""
    fwd = S * exp(-q * T)
    intrinsic = max(fwd - K * exp(-r * T), 0.0) if kind == "call" \
        else max(K * exp(-r * T) - fwd, 0.0)
    if price < intrinsic - 1e-6:
        return np.nan

    def f(sig: float) -> float:
        return bs_price(S, K, T, r, sig, kind, q) - price

    try:
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return float(brentq(f, 1e-6, 5.0, maxiter=100, xtol=1e-8))  # type: ignore[arg-type]
    except ValueError:
        return np.nan


def _calibrate_flat_vol(df: pd.DataFrame, q: float = Q_SPY) -> float:
    """Median ATM implied vol for one expiry's data. ATM band: 0.95 ≤ K/S ≤ 1.05."""
    ivs = [
        implied_vol(float(r["mid"]), float(r["S"]), float(r["K"]),
                    float(r["T"]), float(r["r"]), str(r["kind"]), q)
        for _, r in df.iterrows()
    ]
    df2 = df.copy()
    df2["_iv"] = ivs
    atm = df2[df2["K"].div(df2["S"]).between(0.95, 1.05)]["_iv"].dropna()
    n_atm = len(atm)
    if n_atm < 3:
        atm = df2["_iv"].dropna()
    print(f"      [vol-calib] {n_atm} ATM rows")
    return float(atm.median()) if len(atm) > 0 else 0.3


# ── Entry point 2: build bs_output.csv ───────────────────────────────────────

def run_on_dataset(q: float = Q_SPY) -> None:
    """
    For each expiry: calibrate flat vol from ATM rows, price all rows,
    compute residual = mid - bs_price (target for theory_informed RF).
    """
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    PROC = os.path.join(ROOT, "data", "processed")

    df = pd.read_csv(os.path.join(PROC, "dataset.csv"))
    frames = []
    for expiry, grp in df.groupby("expiry"):
        flat_vol        = _calibrate_flat_vol(grp, q=q)
        grp             = grp.copy()
        grp["flat_vol"] = flat_vol
        grp["bs_price"] = [
            bs_price(float(r["S"]), float(r["K"]), float(r["T"]),
                     float(r["r"]), flat_vol, str(r["kind"]), q)
            for _, r in grp.iterrows()
        ]
        grp["residual"] = grp["mid"] - grp["bs_price"]
        print(f"  {expiry}: flat_vol={flat_vol:.4f}  n={len(grp)}  "
              f"median_resid={grp['residual'].median():.4f}")
        frames.append(grp)

    df = pd.concat(frames, ignore_index=True)
    outp = os.path.join(PROC, "bs_output.csv")
    df.to_csv(outp, index=False)
    print(f"\nWrote {outp}  ({len(df)} rows)")


# ── Entry point 3: flat-vol baseline for compare_models.py ───────────────────

def flat_vol_predict(df_train: pd.DataFrame, df_test: pd.DataFrame,
                     q: float = Q_SPY) -> pd.Series:
    """
    Calibrate flat vol from the nearest training expiry to the test expiry,
    then price all test rows with that vol.

    Using the nearest expiry (rather than all training data) accounts for the
    vol term structure — short-dated expiries have higher IV than long-dated ones.
    """
    test_expiry    = sorted(df_test["expiry"].unique())[-1]
    train_expiries = sorted(df_train["expiry"].unique())
    before         = [e for e in train_expiries if e < test_expiry]
    nearest        = before[-1] if before else train_expiries[-1]

    flat_vol = _calibrate_flat_vol(df_train[df_train["expiry"] == nearest], q=q)
    print(f"  Nearest training expiry : {nearest}")
    print(f"  Flat vol (ATM IV, q={q:.3f}) : {flat_vol:.4f}")

    return pd.Series([
        bs_price(float(r["S"]), float(r["K"]), float(r["T"]),
                 float(r["r"]), flat_vol, str(r["kind"]), q)
        for _, r in df_test.iterrows()
    ], index=df_test.index)


if __name__ == "__main__":
    run_on_dataset()