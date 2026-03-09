"""
brute_force.py  —  Random Forest trained directly on option mid prices

No financial theory used in features. Serves as a data-driven baseline
to compare against the theory-informed model.

Features:
  moneyness     — K/S (linear scale)
  log_moneyness — log(K/S) (log scale, symmetric around ATM)
  T             — time to expiry in years
  delta_proxy   — BS delta at flat_vol (theory-lite, no leakage: doesn't use mid)
  intrinsic     — max(S-K, 0) for calls / max(K-S, 0) for puts

Separate models fitted for calls and puts.
Test set: second-to-last expiry. Last expiry excluded (too far out).
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

FEATURES = ["moneyness", "log_moneyness", "T", "delta_proxy", "intrinsic"]


def load_data() -> pd.DataFrame:
    # Prefer bs_output.csv (has flat_vol for delta_proxy)
    path = os.path.join(PROC, "bs_output.csv")
    if not os.path.exists(path):
        path = os.path.join(PROC, "dataset.csv")
        print("  NOTE: bs_output.csv not found, falling back to dataset.csv")

    df = pd.read_csv(path)
    if "flat_vol" not in df.columns:
        df["flat_vol"] = 0.20   # placeholder so delta_proxy doesn't crash

    df = add_features(df)
    return df.dropna(subset=["mid"] + FEATURES + ["kind", "expiry"])


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expiries = sorted(df["expiry"].unique())
    if len(expiries) < 2:
        raise RuntimeError("Need at least 2 expiries for a temporal split.")

    test_expiry = expiries[-2]   # second-to-last
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

    preds = np.zeros(len(test))
    importances = {}

    for kind in ["call", "put"]:
        tr     = train[train["kind"] == kind]
        te     = test[test["kind"] == kind]
        te_idx = test.index.get_indexer(te.index)

        model = _rf()
        model.fit(tr[FEATURES].to_numpy(), tr["mid"].to_numpy())
        p = model.predict(te[FEATURES].to_numpy())
        preds[te_idx] = p

        y    = te["mid"].to_numpy()
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        mape = float((np.abs(p - y) / np.clip(y, 1e-6, None)).mean() * 100)
        bias = float((p - y).mean())
        print(f"  [{kind}] RMSE={rmse:.4f}  MAPE={mape:.2f}%  Bias={bias:+.4f}  N={len(te)}")
        importances[kind] = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    y_all = test["mid"].to_numpy()
    print(f"\nBrute-force RF — RMSE: {np.sqrt(mean_squared_error(y_all, preds)):.4f} | "
          f"MAPE: {(np.abs(preds - y_all) / np.clip(y_all, 1e-6, None)).mean() * 100:.2f}% | "
          f"Bias: {(preds - y_all).mean():+.4f}")

    for kind in ["call", "put"]:
        print(f"\nFeature importances [{kind}]:")
        for feat, imp in importances[kind].items():
            print(f"  {feat:<18} {imp:.4f}")

    out = test.copy()
    out["mid_true"] = test["mid"].to_numpy()
    out["mid_pred"] = preds
    out.to_csv(os.path.join(PROC, "brute_output.csv"), index=False)
    print(f"Wrote brute_output.csv with {len(out)} rows")


if __name__ == "__main__":
    main()