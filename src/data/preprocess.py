"""
preprocess.py  —  build data/processed/dataset.csv from raw option chain CSVs

Filters applied:
  - Expiries < 30 days skipped: near-expiry prices are unreliable
  - Expiries > 365 days skipped: LEAPS behave differently, distort the split
  - Rows with volume == 0 AND openInterest == 0 dropped (illiquid)
  - Options with mid price < $1.00 dropped (spread often exceeds price)
  - Mid price: bid/ask midpoint where available, lastPrice otherwise
"""

import os
import json

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

MIN_DAYS  = 30
MAX_DAYS  = 365
MIN_PRICE = 1.00   # drop options below $1 — spread is often comparable to price


def load_meta() -> dict:
    with open(os.path.join(RAW, "meta.json")) as f:
        return json.load(f)


def prep_chain(calls: pd.DataFrame, puts: pd.DataFrame,
               S: float, T: float, expiry: str) -> pd.DataFrame:
    frames = []
    for df, kind in [(calls, "call"), (puts, "put")]:
        for col in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest"]:
            if col not in df.columns:
                df[col] = np.nan

        out = df[["strike", "lastPrice", "bid", "ask", "volume", "openInterest"]].copy()
        out["kind"]   = kind
        out["expiry"] = expiry

        # Use bid/ask midpoint where both are available and positive
        has_quote  = (out["bid"].fillna(0) > 0) & (out["ask"].fillna(0) > 0)
        out["mid"] = out["bid"].add(out["ask"]).div(2).where(has_quote, out["lastPrice"])

        out["S"] = S
        out["T"] = T
        out = out.rename(columns={"strike": "K"})

        # Drop illiquid rows
        liquid = (out["volume"].fillna(0) > 0) | (out["openInterest"].fillna(0) > 0)
        n_drop = (~liquid).sum()
        out    = out[liquid]
        if n_drop:
            print(f"    [{kind}] dropped {n_drop} illiquid rows")

        # Drop cheap options — sub-$1 mid prices are unreliable
        out = out[out["mid"].fillna(0) >= MIN_PRICE]

        out = out.dropna(subset=["K", "mid"]).reset_index(drop=True)
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    meta       = load_meta()
    underlying = meta["underlying"]
    S          = float(meta["S"])
    r          = float(meta["r"])
    fetch_date = meta["fetch_date"]
    expiries   = meta["expiries"]

    print(f"Underlying : {underlying}")
    print(f"Spot (S)   : {S:.4f}  (from {fetch_date}, put-call parity)")
    print(f"Risk-free r: {r:.4f}\n")

    frames, n_skipped = [], 0
    for expiry in expiries:
        days = (pd.to_datetime(expiry) - pd.to_datetime(fetch_date)).days
        T    = max(days, 1) / 365.0

        if days < MIN_DAYS:
            print(f"  {expiry}  {days}d — skipped (< {MIN_DAYS} days)")
            n_skipped += 1
            continue
        if days > MAX_DAYS:
            print(f"  {expiry}  {days}d — skipped (> {MAX_DAYS} days, LEAPS)")
            n_skipped += 1
            continue

        calls_path = os.path.join(RAW, f"options_calls_{underlying}_{expiry}.csv")
        puts_path  = os.path.join(RAW, f"options_puts_{underlying}_{expiry}.csv")
        if not os.path.exists(calls_path) or not os.path.exists(puts_path):
            print(f"  {expiry} — WARNING: CSV not found, skipping")
            continue

        frame = prep_chain(pd.read_csv(calls_path), pd.read_csv(puts_path), S, T, expiry)
        frames.append(frame)
        print(f"  {expiry}  T={T:.3f}yr  kept {len(frame)} rows")

    if not frames:
        raise RuntimeError("No data loaded — run loader.py first.")
    print(f"\n  ({n_skipped} expiries skipped)")

    df = pd.concat(frames, ignore_index=True)
    df["r"] = r
    df = df[["expiry", "K", "kind", "mid", "S", "T", "r", "volume", "openInterest"]]

    out_path = os.path.join(PROC, "dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df)} rows, {df['kind'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()