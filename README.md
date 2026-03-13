# SPY Options Pricing: Black-Scholes vs Machine Learning

An end-to-end pipeline that fetches live SPY option chains, prices them with three models of increasing sophistication, and evaluates them on a held-out expiry.

The central question: can a machine learning model learn and correct for the systematic mispricing caused by Black-Scholes' constant volatility assumption?

## Table of Contents
1. [Motivation](#motivation)
2. [Assumptions & Design Choices](#assumptions--design-choices)
3. [Data Pipeline](#data-pipeline)
4. [Model Architecture](#model-architecture)
5. [Results & Limitations](#results--limitations)
6. [Project Structure](#project-structure)
7. [Setup & Usage](#setup--usage)
8. [References](#references)

## Motivation

Black-Scholes prices options under the assumption that volatility is constant across strikes and expiries. In reality, markets exhibit a volatility smile: out-of-the-money puts trade at higher implied volatility than at-the-money options, and out-of-the-money calls trade lower. This is a well-documented, systematic failure.

The question this project asks is simple: given a snapshot of the SPY options market, can we train a model that learns this smile and prices options more accurately than a flat-vol Black-Scholes baseline?

## Assumptions & Design Choices

These are the key decisions made during the project and the reasoning behind each one.

**Underlying: SPY instead of commodity futures.** SPY was chosen because SPY options are among the most liquid in the world. Liquid options have tight bid/ask spreads, meaning the mid price is a reliable estimate of fair value. Illiquid options like USO often have stale last prices and wide spreads that would contaminate the training signal.

**Spot price derived from put-call parity.** Rather than using yfinance's potentially stale last price for the underlying, spot price S is derived from put-call parity across ATM strike pairs on the nearest liquid expiry:

```
S = (C − P + K · exp(−r · T)) / exp(−q · T)
```

The median across valid strike pairs is used as the effective spot. This is the price the options market itself is implying, which is the right reference for pricing consistency.

**Merton formula with continuous dividend yield.** SPY pays a continuous dividend yield of approximately 1.3%. Ignoring this would systematically underprice calls and overprice puts. The Merton (1973) extension of Black-Scholes accounts for this:

```
d1 = [log(S/K) + (r − q + σ²/2) · T] / (σ · √T)
d2 = d1 − σ · √T
```

**Mid price as the target.** Where both bid and ask are available and positive, mid price = (bid + ask) / 2. This is the standard proxy for fair value used by practitioners. If only last price is available, it is used as fallback.

**Liquidity filters.** Several filters are applied to remove unreliable data points. Options with zero volume and zero open interest are dropped since they are not actively traded. Options with a mid price below $1.00 are also dropped because the bid/ask spread is often comparable to the price itself, making the mid a noisy estimate. Expiries under 30 days are excluded due to near-expiry microstructure effects, and expiries over 365 days are excluded because LEAPS behave differently.

**Flat vol calibrated per expiry.** For each expiry, a single implied volatility is extracted from ATM options where moneyness is within 2% of 1. This flat vol is used as the Black-Scholes input for that expiry. It captures the level of vol without making assumptions about the smile shape, which is what the ML models are there to learn.

**Temporal train/test split.** Training uses all expiries before the test expiry, and evaluation is done on a single held-out expiry. This prevents any form of lookahead, since the models never see future information during training.

## Data Pipeline

```
loader.py → preprocess.py → models → compare_models.py
```

**Step 1: Fetch (loader.py).** Downloads the full SPY option chain via yfinance for all available expiries, filters to expiries between 30 and 365 days, derives spot price S from put-call parity, and writes meta.json with S, r, q, and the fetch date.

**Step 2: Preprocess (preprocess.py).** Computes mid price from bid/ask where available, drops illiquid and sub-$1 options, adds time to expiry T in years, and writes the clean dataset to `data/processed/dataset.csv`.

**Step 3: Price (models).** Three models run independently on the same dataset and each writes its predictions to `data/processed/`.

**Step 4: Evaluate (compare_models.py).** Merges all three model outputs with market prices, computes RMSE, MAPE, bias, and R² overall and by moneyness bucket, and prints a structured report.

## Model Architecture

**Model 1: Flat-vol Black-Scholes (baseline).** The classical Merton formula with per-expiry ATM implied vol. No learning, no data, pure theory. It serves as the benchmark everything else is measured against. Its failure mode is well-known: by assuming a flat vol surface, it systematically misprices OTM options where the smile is most pronounced.

**Model 2: Brute-force Random Forest.** A standard Random Forest trained directly on option mid prices, with no financial structure imposed. Separate models are trained for calls and puts. Features are moneyness, log-moneyness, time to expiry, delta proxy (BS delta at flat vol), and intrinsic value. The main weakness is the absence of a financial prior: the model struggles on deep ITM options where training data is sparse and tends to regress toward the mean, producing a systematic negative bias.

**Model 3: Theory-informed Random Forest.** Instead of predicting the option price directly, this model predicts the residual between the market price and the BS price:

```
residual = mid_price − bs_price
final_prediction = bs_price + predicted_residual
```

This is the key insight. BS already gets the price level approximately right, so the RF only needs to learn the smile correction, which is a much simpler and better-conditioned problem. The same features as the brute-force model are used, with flat vol added as an input. By anchoring on BS, the model inherits the correct scale and moneyness structure for free, and the residual it needs to learn is smaller and smoother than the raw price.

## Results & Limitations

### Overall performance (test expiry: 2026-12-31, N=272)

| Model | RMSE ($) | MAPE | Bias ($) | R² |
|---|---|---|---|---|
| Black-Scholes (flat vol) | 7.80 | 46.9% | +0.76 | 0.957 |
| Brute-force RF | 6.62 | 11.9% | -5.21 | 0.969 |
| Theory-informed RF | **3.26** | **4.9%** | +0.25 | **0.992** |

### By moneyness bucket (Theory-informed RF)

| Bucket | RMSE ($) | MAPE | N |
|---|---|---|---|
| OTM calls (K/S > 1.02) | 1.29 | 11.7% | 72 |
| ATM calls (0.98–1.02) | 1.84 | 3.3% | 23 |
| ITM calls (K/S < 0.98) | 6.60 | 4.3% | 48 |
| OTM puts (K/S < 0.98) | 1.13 | 1.6% | 57 |
| ATM puts (0.98–1.02) | 1.67 | 2.2% | 25 |
| ITM puts (K/S > 1.02) | 3.14 | 1.7% | 47 |

The largest gains over BS are on OTM options, which is exactly where vol skew causes the most systematic mispricing. ITM options are harder because intrinsic value dominates and the smile correction is smaller, making BS more competitive in that region.

Feature importances confirm that delta proxy is the dominant predictor in both call and put models, reflecting that the joint moneyness and time structure is the most informative input for pricing.

### Known limitations

**Single cross-section.** All data was fetched on a single day. The temporal split by expiry tests vol surface interpolation across maturities, not true out-of-sample generalisation across time. A model trained on March 2026 data may not perform equally well in a different market regime.

**Brute-force systematic bias.** The -$5.21 bias comes from sparse training data on deep ITM options. The model regresses toward the mean of a training set dominated by OTM and ATM options.

**R² is inflated by price variance.** The dataset spans a large price range, from $1 OTM options to $100+ deep ITM options. R² looks high partly because of this variance, which is why RMSE in dollars is the more honest metric.

**MAPE is only comparable within buckets.** A given dollar error produces a higher MAPE on a cheap OTM option than on an expensive ITM option. Comparing MAPE across moneyness buckets is misleading.

## Project Structure

```
options-pricing-ml/
├── src/
│   └── data/
│       ├── loader.py               # Fetch SPY chains, derive S from PCP
│       ├── preprocess.py           # Filter and build dataset.csv
│       ├── models/
│       │   ├── features.py         # Shared feature engineering
│       │   ├── black_scholes.py    # BS pricer, IV solver, flat-vol calibration
│       │   ├── brute_force.py      # Brute-force Random Forest
│       │   └── theory_informed.py  # Theory-informed Random Forest
│       └── evaluation/
│           └── compare_models.py   # 3-way evaluation and metrics
├── run_pipeline.py                 # Orchestrates all steps
├── data/
│   ├── raw/                        # Fetched option chains (git-ignored)
│   └── processed/                  # Pipeline outputs (git-ignored)
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

## References

- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Merton, R. C. (1973). Theory of rational option pricing. *Bell Journal of Economics*, 4(1), 141–183.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
