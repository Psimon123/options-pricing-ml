"""
theory_informed.py  —  RF trained on Black-Scholes residuals

Architecture: predict residual = mid - bs_price(flat_vol), then
reconstruct the final price as bs_price + predicted_residual.

This separates what BS gets right (level, rough shape) from what it
gets wrong (vol skew, smile). The RF only needs to learn the correction.

Features:
  flat_vol      — vol level used for this expiry's BS price
  moneyness     — K/S
  log_moneyness — log(K/S)
  T             — time to expiry in years
  delta_proxy   — BS delta at flat_vol (encodes moneyness+T jointly, no leakage)
  intrinsic     — max(S-K, 0) for calls / max(K-S, 0) for puts

Requires bs_output.csv from black_scholes.py.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from features import add_features  # type: ignore[import]

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")

FEATURES = ["flat_vol", "moneyness", "log_moneyness", "T", "delta_proxy", "intrinsic"]


def load_data() -> pd.DataFrame:
    path = os.path.join(PROC, "bs_output.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run black_scholes.py first.")

    df = pd.read_csv(path)
    df = add_features(df)
    return df.dropna(subset=["mid", "bs_price", "residual", "flat_vol"] + FEATURES + ["kind", "expiry"])


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expiries = sorted(df["expiry"].unique())
    if len(expiries) < 2:
        raise RuntimeError("Need at least 2 expiries for a temporal split.")

    test_expiry = expiries[-2]
    train = df[df["expiry"] < test_expiry].copy()
    test  = df[df["expiry"] == test_expiry].copy()

    print(f"  Train expiries : {sorted(df[df['expiry'] < test_expiry]['expiry'].unique())}")
    print(f"  Test expiry    : {test_expiry}")
    print(f"  (Last expiry {expiries[-1]} excluded)")
    print(f"  Train rows: {len(train)}  |  Test rows: {len(test)}")
    return train, test


def _rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400, max_depth=12,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
    )


def main() -> None:
    df = load_data()
    train, test = temporal_split(df)

    resid_preds = np.zeros(len(test))
    price_preds = np.zeros(len(test))
    importances = {}

    for kind in ["call", "put"]:
        tr     = train[train["kind"] == kind]
        te     = test[test["kind"] == kind]
        te_idx = test.index.get_indexer(te.index)

        model = _rf()
        model.fit(tr[FEATURES].to_numpy(), tr["residual"].to_numpy())
        rp = model.predict(te[FEATURES].to_numpy())
        pp = te["bs_price"].to_numpy() + rp

        resid_preds[te_idx] = rp
        price_preds[te_idx] = pp

        y    = te["mid"].to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y, pp)))
        mape = float((np.abs(pp - y) / np.clip(y, 1e-6, None)).mean() * 100)
        bias = float((pp - y).mean())
        print(f"  [{kind}] RMSE={rmse:.4f}  MAPE={mape:.2f}%  Bias={bias:+.4f}  N={len(te)}")
        importances[kind] = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    y_all = test["mid"].to_numpy()
    print(f"\nTheory-informed RF — RMSE: {np.sqrt(mean_squared_error(y_all, price_preds)):.4f} | "
          f"MAPE: {(np.abs(price_preds - y_all) / np.clip(y_all, 1e-6, None)).mean() * 100:.2f}% | "
          f"Bias: {(price_preds - y_all).mean():+.4f}")

    for kind in ["call", "put"]:
        print(f"\nFeature importances [{kind}]:")
        for feat, imp in importances[kind].items():
            print(f"  {feat:<18} {imp:.4f}")

    out = test.copy()
    out["residual_pred"] = resid_preds
    out["price_pred"]    = price_preds
    out["mid_true"]      = test["mid"].to_numpy()
    out.to_csv(os.path.join(PROC, "theory_output.csv"), index=False)
    print(f"Wrote theory_output.csv with {len(out)} rows")


if __name__ == "__main__":
    main()