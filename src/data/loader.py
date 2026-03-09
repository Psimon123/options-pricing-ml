"""
loader.py  —  fetch SPY option chains and save to data/raw/

Spot price derived from put-call parity rather than yfinance price feeds,
which can lag intraday moves. For a pair of call/put at the same strike K
and expiry T, put-call parity gives:

    S = (C - P + K·exp(-r·T)) / exp(-q·T)

Solving across all valid strike pairs on the nearest liquid expiry and
taking the median gives the effective spot price the options market is using.
"""

import os
import json
from math import exp
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW  = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)

OPTION_TICKER  = "SPY"
FUTURES_TICKER = "ES=F"
R_DEFAULT      = 0.053   # 3-month T-bill rate
Q_DEFAULT      = 0.013   # SPY continuous dividend yield (~1.3% as of 2026)


def _derive_S_from_pcp(calls: pd.DataFrame, puts: pd.DataFrame,
                        expiry: str, fetch_date: str,
                        r: float, q: float) -> Optional[float]:
    """
    Derive effective spot price from put-call parity on one expiry.
    Returns median implied S across valid strike pairs, or None if < 3 found.
    """
    T = (pd.to_datetime(expiry) - pd.to_datetime(fetch_date)).days / 365.0
    if T <= 0:
        return None

    c_prices = calls.set_index("strike")["lastPrice"]
    p_prices = puts.set_index("strike")["lastPrice"]

    implied = []
    for K in c_prices.index.intersection(p_prices.index):
        c, p = float(c_prices.at[K]), float(p_prices.at[K])
        if c > 0.10 and p > 0.10:
            S = (c - p + K * exp(-r * T)) / exp(-q * T)
            if S > 0:
                implied.append(S)

    return float(np.median(implied)) if len(implied) >= 3 else None


def main() -> None:
    # Futures reference data
    hist = yf.Ticker(FUTURES_TICKER).history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No price history for {FUTURES_TICKER}")
    hist.to_csv(os.path.join(RAW, "cl_futures.csv"))
    print(f"Saved futures ({FUTURES_TICKER}) -> data/raw/cl_futures.csv")

    # Fetch date from last market session
    fetch_date = pd.Timestamp(hist.index[-1]).normalize().strftime("%Y-%m-%d")
    print(f"\nFetch date : {fetch_date}")
    print(f"Fetching option chains for {OPTION_TICKER} and deriving S from PCP...")

    ticker   = yf.Ticker(OPTION_TICKER)
    expiries = ticker.options
    if not expiries:
        raise RuntimeError(f"No option expiries available for {OPTION_TICKER}")

    S, S_expiry = None, None
    chains = {}

    print(f"  Fetching {len(expiries)} expiries...")
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        chains[expiry] = chain

        if S is None:
            candidate = _derive_S_from_pcp(
                chain.calls, chain.puts, expiry, fetch_date, R_DEFAULT, Q_DEFAULT
            )
            if candidate is not None:
                S, S_expiry = candidate, expiry
                print(f"  S = {S:.4f}  (put-call parity on {S_expiry})")

        print(f"  Saved {OPTION_TICKER} chain -> expiry {expiry}")

    if S is None:
        raise RuntimeError(
            "Could not derive S from put-call parity on any expiry. "
            "Check that SPY options have valid bid/ask quotes."
        )

    # Save per-expiry CSVs
    for expiry, chain in chains.items():
        chain.calls.to_csv(os.path.join(RAW, f"options_calls_{OPTION_TICKER}_{expiry}.csv"), index=False)
        chain.puts.to_csv(os.path.join(RAW,  f"options_puts_{OPTION_TICKER}_{expiry}.csv"),  index=False)

    meta = {
        "underlying":  OPTION_TICKER,
        "futures_ref": FUTURES_TICKER,
        "fetch_date":  fetch_date,
        "S":           S,
        "S_method":    f"put_call_parity:{S_expiry}",
        "r":           R_DEFAULT,
        "q":           Q_DEFAULT,
        "expiries":    list(expiries),
    }
    meta_path = os.path.join(RAW, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()