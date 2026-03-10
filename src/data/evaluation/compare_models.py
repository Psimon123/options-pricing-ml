"""
compare_models.py  —  3-way evaluation: flat-vol BS vs brute RF vs theory-informed RF

All three models evaluated on the same test rows (second-to-last expiry).

Moneyness buckets are split by kind to avoid the misleading case where
K/S > 1.1 mixes deep-OTM calls (worth pennies, huge MAPE) with deep-ITM
puts (worth hundreds). RMSE in dollars is the primary comparison metric;
MAPE is meaningful only within the same kind+moneyness bucket.
"""

import os
import sys
import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
sys.path.insert(0, MODELS)

from black_scholes import flat_vol_predict  # type: ignore[import]

ROOT     = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROC     = os.path.join(ROOT, "data", "processed")
MERGE_KEY = ["K", "kind", "expiry"]


def metrics(df: pd.DataFrame) -> dict:
    err   = df["pred"] - df["mid_true"]
    abs_e = err.abs()
    ss_res = (err ** 2).sum()
    ss_tot = ((df["mid_true"] - df["mid_true"].mean()) ** 2).sum()
    return {
        "rmse":    float(np.sqrt((err ** 2).mean())),
        "mape":    float((abs_e / np.clip(df["mid_true"], 1e-6, None)).mean() * 100),
        "bias":    float(err.mean()),
        "max_err": float(abs_e.max()),
        "r2":      float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n":       int(len(df)),
    }


def print_metrics(label: str, m: dict) -> None:
    print(f"  {label:<26} RMSE=${m['rmse']:.2f}  MAPE={m['mape']:.1f}%  "
          f"Bias={m['bias']:+.2f}  R²={m['r2']:.4f}  N={m['n']}")


def bucket_report(label: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in ["call", "put"]:
        sub = df[df["kind"] == kind]
        if kind == "call":
            buckets = [
                ("OTM call (K/S > 1.02)", sub[sub["moneyness"] > 1.02]),
                ("ATM call (0.98–1.02)",   sub[sub["moneyness"].between(0.98, 1.02)]),
                ("ITM call (K/S < 0.98)",  sub[sub["moneyness"] < 0.98]),
            ]
        else:
            buckets = [
                ("OTM put  (K/S < 0.98)", sub[sub["moneyness"] < 0.98]),
                ("ATM put  (0.98–1.02)",   sub[sub["moneyness"].between(0.98, 1.02)]),
                ("ITM put  (K/S > 1.02)",  sub[sub["moneyness"] > 1.02]),
            ]
        for name, grp in buckets:
            if len(grp) >= 3:
                rows.append({"model": label, "bucket": name, **metrics(grp)})
    return pd.DataFrame(rows)


def load_all():
    brute_raw = pd.read_csv(os.path.join(PROC, "brute_output.csv"))
    theor_raw = pd.read_csv(os.path.join(PROC, "theory_output.csv"))

    exp_b = sorted(brute_raw["expiry"].unique())[-1]
    exp_t = sorted(theor_raw["expiry"].unique())[-1]
    if exp_b != exp_t:
        raise RuntimeError(f"Test expiry mismatch: brute={exp_b}, theory={exp_t}. Re-run both models.")
    test_expiry = exp_b
    print(f"Test expiry : {test_expiry}")

    brute = brute_raw[MERGE_KEY + ["S", "mid_true"]].copy()
    brute["pred"] = brute_raw["mid_pred"]

    theor = theor_raw[MERGE_KEY + ["S", "mid_true"]].copy()
    theor["pred"] = theor_raw["price_pred"]

    # Align to common rows only
    common = brute[MERGE_KEY].merge(theor[MERGE_KEY], on=MERGE_KEY, how="inner")
    brute  = brute.merge(common, on=MERGE_KEY, how="inner").reset_index(drop=True)
    theor  = theor.merge(common, on=MERGE_KEY, how="inner").reset_index(drop=True)
    print(f"Common rows : {len(brute)}")

    # Build BS baseline
    dataset  = pd.read_csv(os.path.join(PROC, "dataset.csv"))
    df_train = dataset[dataset["expiry"] != test_expiry].copy()
    df_test  = dataset[dataset["expiry"] == test_expiry].merge(common, on=MERGE_KEY, how="inner").reset_index(drop=True)

    print("\nCalibrating flat-vol BS baseline...")
    bs_preds = flat_vol_predict(df_train, df_test)
    bs = pd.DataFrame({
        "K": df_test["K"], "S": df_test["S"],
        "kind": df_test["kind"], "expiry": df_test["expiry"],
        "mid_true": df_test["mid"], "pred": bs_preds.values,
    })

    for df in (bs, brute, theor):
        df["moneyness"] = df["K"] / df["S"]

    return bs, brute, theor


def main() -> None:
    bs, brute, theor = load_all()

    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    m_bs, m_br, m_th = metrics(bs), metrics(brute), metrics(theor)
    print_metrics("Black-Scholes (flat vol)", m_bs)
    print_metrics("Brute-force RF", m_br)
    print_metrics("Theory-informed RF", m_th)

    print("\n" + "=" * 70)
    print("BY KIND × MONEYNESS  (RMSE in $, MAPE meaningful within-bucket only)")
    print("=" * 70)
    bkt = pd.concat([
        bucket_report("BS", bs),
        bucket_report("Brute", brute),
        bucket_report("Theory", theor),
    ])
    print(bkt[["model", "bucket", "rmse", "mape", "bias", "n"]].to_string(index=False))

    summary = pd.DataFrame([
        {"model": "black_scholes_flat_vol", **m_bs},
        {"model": "brute_force_rf", **m_br},
        {"model": "theory_informed_rf", **m_th},
    ])
    summary.to_csv(os.path.join(PROC, "metrics_summary.csv"), index=False)
    print(f"\nWrote metrics_summary.csv")


if __name__ == "__main__":
    main()