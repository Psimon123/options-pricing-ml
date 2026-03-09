# SPY Options Pricing: Black-Scholes vs Machine Learning

An end-to-end pipeline that fetches live SPY option chains, prices them with three models of increasing sophistication, and evaluates them on a held-out expiry.

## Motivation

Black-Scholes assumes constant volatility across strikes — a known failure. Real markets exhibit a **volatility smile/skew**: OTM puts trade at higher implied vol than ATM options, and OTM calls trade lower. The question this project asks: can a machine learning model learn and correct for this systematic BS mispricing?

## Models

| Model | Description |
|---|---|
| **Flat-vol Black-Scholes** | Merton (1973) formula with continuous dividend yield. Flat vol calibrated per-expiry from ATM implied vols. Serves as the theory baseline. |
| **Brute-force RF** | Random Forest trained directly on option mid prices. No financial theory — pure data-driven baseline. |
| **Theory-informed RF** | RF trained on BS *residuals* (mid − BS price), then adds back the BS price. Separates what BS gets right from what it gets wrong. |

## Results

Test set: held-out expiry 2026-12-31 (272 options, temporal split — no lookahead).

### Overall

| Model | RMSE ($) | MAPE | Bias ($) | R² |
|---|---|---|---|---|
| Black-Scholes (flat vol) | 7.80 | 46.9% | +0.76 | 0.957 |
| Brute-force RF | 6.62 | 11.9% | −5.21 | 0.969 |
| Theory-informed RF | **3.26** | **4.9%** | +0.25 | **0.992** |

### By moneyness bucket (Theory-informed RF)

| Bucket | RMSE ($) | MAPE |
|---|---|---|
| OTM calls (K/S > 1.02) | 1.29 | 11.7% |
| ATM calls (0.98–1.02) | 1.84 | 3.3% |
| ITM calls (K/S < 0.98) | 6.60 | 4.3% |
| OTM puts (K/S < 0.98) | 1.13 | 1.6% |
| ATM puts (0.98–1.02) | 1.67 | 2.2% |
| ITM puts (K/S > 1.02) | 3.14 | 1.7% |

The largest gains over BS are on OTM options — exactly where vol skew causes the most systematic mispricing.

## Data & Methodology

**Spot price derivation** — Rather than using yfinance's potentially stale last price, spot is derived from put-call parity across ATM strike pairs on the nearest liquid expiry:

```
S = (C − P + K·exp(−r·T)) / exp(−q·T)
```

Median across valid pairs gives the effective spot the options market is pricing.

**Filters applied**
- Expiries < 30 days excluded (near-expiry micro-structure noise)
- Expiries > 365 days excluded (LEAPS behave differently)
- Options with zero volume AND zero open interest dropped (illiquid)
- Options with mid price < $1.00 dropped (bid/ask spread often exceeds price)
- Mid price = bid/ask midpoint where both quotes available, else last price

**Training / test split** — Temporal: train on all expiries before the test expiry, evaluate on a single held-out expiry. Last expiry excluded entirely to avoid any forward-looking contamination.

**Features (RF models)**
- `moneyness` — K/S
- `log_moneyness` — log(K/S), symmetric around ATM
- `T` — time to expiry in years
- `delta_proxy` — BS delta at flat vol (encodes moneyness × T jointly, no leakage)
- `intrinsic` — max(S−K, 0) for calls, max(K−S, 0) for puts
- `flat_vol` — per-expiry ATM implied vol (theory-informed model only)

**Known limitations**
- Single cross-section: all data fetched on one day. The temporal split by expiry tests vol surface interpolation, not true multi-day generalisation.
- Brute-force RF shows systematic negative bias on ITM options — sparse training signal in that region causes mean reversion toward ATM prices.
- MAPE is misleading across moneyness buckets; RMSE ($) is the primary metric.

## Project Structure

```
options-pricing-ml/
├── src/
│   └── data/
│       ├── loader.py           # Fetch SPY chains, derive S from PCP
│       ├── preprocess.py       # Filter and build dataset.csv
│       ├── models/
│       │   ├── features.py         # Shared feature engineering
│       │   ├── black_scholes.py    # BS pricer, IV solver, flat-vol calibration
│       │   ├── brute_force.py      # Brute-force Random Forest
│       │   └── theory_informed.py  # Theory-informed Random Forest
│       └── evaluation/
│           └── compare_models.py   # 3-way evaluation and metrics
├── run_pipeline.py             # Orchestrates all steps
├── data/
│   ├── raw/                    # Fetched option chains (git-ignored)
│   └── processed/              # Pipeline outputs (git-ignored)
├── pyrightconfig.json
├── requirements.txt
└── README.md
```

## Setup & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (fetch + price + evaluate)
python run_pipeline.py

# Resume from a specific step
python run_pipeline.py --from preprocess
python run_pipeline.py --from bs
python run_pipeline.py --from compare
```

## Requirements

```
yfinance
pandas
numpy
scipy
scikit-learn
```

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Merton, R. C. (1973). Theory of rational option pricing. *Bell Journal of Economics*, 4(1), 141–183.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
