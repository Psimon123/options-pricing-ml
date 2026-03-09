"""
run_pipeline.py  —  end-to-end options pricing pipeline runner

Steps:
  1. loader          — fetch SPY option chains from yfinance, save raw CSVs
  2. preprocess      — build data/processed/dataset.csv
  3. black_scholes   — calibrate flat vol per expiry, build bs_output.csv
  4. brute_force     — train brute-force RF, build brute_output.csv
  5. theory_informed — train theory-informed RF, build theory_output.csv
  6. compare_models  — 3-way evaluation, print metrics, save metrics_summary.csv

Usage:
  python run_pipeline.py              # run all steps
  python run_pipeline.py --from bs    # resume from black_scholes onward
  python run_pipeline.py --only compare  # run a single step

  --fresh   wipe all cached data and start clean
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

# ── project root (where this file lives) ─────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── step registry ─────────────────────────────────────────────────────────────
# Each step: (short_name, module_path_relative_to_ROOT)
STEPS = [
    ("loader",         "src/data/loader.py"),
    ("preprocess",     "src/data/preprocess.py"),
    ("bs",             "src/data/models/black_scholes.py"),
    ("brute",          "src/data/models/brute_force.py"),
    ("theory",         "src/data/models/theory_informed.py"),
    ("compare",        "src/data/evaluation/compare_models.py"),
]

STEP_NAMES = [name for name, _ in STEPS]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_and_run(name: str, rel_path: str) -> None:
    """Dynamically import a module by file path and call its main() function."""
    abs_path = os.path.join(ROOT, rel_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Module not found: {abs_path}")

    spec = importlib.util.spec_from_file_location(name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {abs_path}")
    module = importlib.util.module_from_spec(spec)

    # Add the module's directory to sys.path so relative imports work
    module_dir = os.path.dirname(abs_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec.loader.exec_module(module)

    # Different modules use different entry point names
    if hasattr(module, "main"):
        module.main()
    elif hasattr(module, "build_processed"):
        module.build_processed()
    elif hasattr(module, "run_on_dataset"):
        module.run_on_dataset()
    else:
        raise AttributeError(
            f"{rel_path} has no recognised entry point "
            "(expected main(), build_processed(), or run_on_dataset())"
        )


def _wipe_data() -> None:
    """Delete all cached raw and processed data files."""
    raw_dir  = os.path.join(ROOT, "data", "raw")
    proc_dir = os.path.join(ROOT, "data", "processed")
    count    = 0
    for d in (raw_dir, proc_dir):
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith((".csv", ".json")):
                os.remove(os.path.join(d, fname))
                count += 1
    print(f"  Wiped {count} cached data files.\n")


def _banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP: {text}")
    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run the options pricing pipeline.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_step", metavar="STEP",
        choices=STEP_NAMES,
        help=f"Resume pipeline from this step. Choices: {STEP_NAMES}"
    )
    group.add_argument(
        "--only", dest="only_step", metavar="STEP",
        choices=STEP_NAMES,
        help="Run only this single step."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Wipe all cached data before running."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which steps to run
    if args.only_step:
        steps_to_run = [(n, p) for n, p in STEPS if n == args.only_step]
    elif args.from_step:
        start_idx    = STEP_NAMES.index(args.from_step)
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = STEPS

    if args.fresh:
        print("Wiping cached data...")
        _wipe_data()

    total_start = time.time()
    failed      = False

    for name, path in steps_to_run:
        _banner(name)
        t0 = time.time()
        try:
            _load_and_run(name, path)
            elapsed = time.time() - t0
            print(f"\n  ✓ {name} completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            traceback.print_exc()
            failed = True
            break

    total = time.time() - total_start
    print("\n" + "=" * 60)
    if failed:
        print(f"  Pipeline FAILED after {total:.1f}s")
        sys.exit(1)
    else:
        print(f"  Pipeline complete in {total:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()