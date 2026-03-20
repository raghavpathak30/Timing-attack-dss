"""
attack.py — Correlation-based timing side-channel attack on DSS-like modular
            arithmetic.

Usage examples
--------------
# Generate a fresh dataset and immediately run the attack:
    python attack.py

# Customise key length and dataset size:
    python attack.py --key-bits 8 --samples 10000

# Load a previously saved dataset instead of generating a new one:
    python attack.py --load-dataset dataset/sample_data.csv

# Suppress visualisation:
    python attack.py --no-plot

Attack methodology
------------------
1. A random 16-bit secret key ``x`` is drawn.
2. The "oracle" (vulnerable_mod_mult) computes ``x * r mod q`` for many random
   nonces ``r``, recording execution time for each.
3. For each candidate key ``x_guess`` we compute the *predicted* number of
   reduction iterations (floor(x_guess * r / q)) across all nonces.
4. We compute the Pearson correlation between the predicted-cost vector and the
   observed-timing vector.
5. The guess that maximises the absolute correlation is the recovered key.

When the correct guess is tested, predicted cost and observed timing are
maximally correlated because the predicted model perfectly describes the
leaking computation.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from utils import (
    DEFAULT_KEY_BITS,
    DEFAULT_N_SAMPLES,
    DEFAULT_Q,
    NOISE_STD,
    generate_dataset,
    generate_secret_key,
    load_dataset,
    predicted_timing_cost,
    save_dataset,
)
from visualize import plot_correlation_vs_guesses, plot_timing_histogram


# ---------------------------------------------------------------------------
# Attack core
# ---------------------------------------------------------------------------

def compute_correlations(
    df: pd.DataFrame,
    q: int,
    key_range: range,
) -> dict[int, float]:
    """Compute Pearson correlation between predicted cost and observed timing.

    For each candidate key ``x_guess`` in ``key_range`` we build a vector of
    predicted iteration counts (one entry per sample) and correlate it with the
    observed timing vector.  The true key produces the highest correlation.

    Parameters
    ----------
    df:
        Dataset with columns ``r`` and ``timing_ns``.
    q:
        Group order (prime).
    key_range:
        Iterable of integer key candidates to test.

    Returns
    -------
    dict[int, float]
        Mapping from each ``x_guess`` to its Pearson correlation coefficient.
    """
    timings: np.ndarray = df["timing_ns"].values.astype(float)
    nonces: np.ndarray = df["r"].values.astype(int)

    correlations: dict[int, float] = {}

    for x_guess in key_range:
        predicted: np.ndarray = np.array(
            [predicted_timing_cost(x_guess, r, q) for r in nonces],
            dtype=float,
        )
        # Pearson correlation (handle zero-variance edge cases gracefully)
        if np.std(predicted) == 0 or np.std(timings) == 0:
            correlations[x_guess] = 0.0
        else:
            corr_matrix = np.corrcoef(predicted, timings)
            correlations[x_guess] = float(corr_matrix[0, 1])

    return correlations


def recover_key(correlations: dict[int, float]) -> int:
    """Return the key guess with the highest absolute Pearson correlation.

    Parameters
    ----------
    correlations:
        Output of :func:`compute_correlations`.

    Returns
    -------
    int
        Recovered key estimate.
    """
    return max(correlations, key=lambda g: abs(correlations[g]))


def run_attack(
    df: pd.DataFrame,
    secret_key: int | None,
    q: int = DEFAULT_Q,
    verbose: bool = True,
    plot: bool = True,
) -> int:
    """Execute the full correlation timing attack pipeline.

    Parameters
    ----------
    df:
        Timing measurement dataset.
    secret_key:
        The true key (used for accuracy reporting and plot annotation).
        Pass ``None`` if the true key is unknown.
    q:
        Group order.
    verbose:
        Print progress messages.
    plot:
        Generate and save visualisation plots.

    Returns
    -------
    int
        The recovered (guessed) key value.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  DSS Timing Side-Channel Attack")
        print("=" * 60)
        print(f"  Group order q      : {q}")
        print(f"  Dataset size       : {len(df)} samples")
        if secret_key is not None:
            print(f"  True secret key    : {secret_key}")
        print("=" * 60 + "\n")

    # --- Step 1: Visualise the timing distribution -------------------------
    if plot:
        plot_timing_histogram(df)

    # --- Step 2: Correlation sweep over all candidate keys ----------------
    key_range = range(1, q)
    total = len(key_range)

    if verbose:
        print(f"[*] Testing {total} key candidates …")
    correlations = _vectorised_correlations(df, q, key_range, verbose=verbose)

    # --- Step 3: Pick the best candidate ----------------------------------
    recovered_key = recover_key(correlations)

    # --- Step 4: Report results -------------------------------------------
    if verbose:
        best_corr = correlations[recovered_key]
        print(f"\n[+] Recovered key  : {recovered_key}")
        if secret_key is not None:
            print(f"[+] True key       : {secret_key}")
            match = "✓ CORRECT" if recovered_key == secret_key else "✗ INCORRECT"
            print(f"[+] Match          : {match}")
        print(f"[+] Best |corr|    : {abs(best_corr):.6f}")

    # --- Step 5: Plot correlation landscape --------------------------------
    if plot:
        plot_correlation_vs_guesses(
            correlations,
            true_key=secret_key,
        )

    return recovered_key


# ---------------------------------------------------------------------------
# Vectorised correlation helper (performance optimisation)
# ---------------------------------------------------------------------------

def _vectorised_correlations(
    df: pd.DataFrame,
    q: int,
    key_range: range,
    verbose: bool = True,
) -> dict[int, float]:
    """Compute all correlations using a fully vectorised numpy approach.

    Instead of a Python loop over nonces per key-guess, we build the entire
    prediction matrix at once:

        P[i, j] = floor(key_range[i] * nonces[j] / q)

    then correlate each row with the timing vector in one pass.

    Parameters
    ----------
    df:
        Dataset with ``r`` and ``timing_ns`` columns.
    q:
        Group order.
    key_range:
        Range of key candidates.
    verbose:
        Print progress updates.

    Returns
    -------
    dict[int, float]
        Correlation for each candidate key.
    """
    timings: np.ndarray = df["timing_ns"].values.astype(np.float64)
    nonces: np.ndarray = df["r"].values.astype(np.int64)

    keys: np.ndarray = np.array(list(key_range), dtype=np.int64)   # shape (K,)

    # Predicted iteration count matrix: shape (K, N)
    # Each row = floor(keys[i] * nonces / q)
    # We process in chunks to avoid OOM on large key spaces.
    CHUNK = 1024
    n_keys = len(keys)

    t_mean: float = timings.mean()
    t_std: float = timings.std()
    if t_std == 0:
        return {int(k): 0.0 for k in keys}

    t_norm = timings - t_mean   # shape (N,)

    correlations: dict[int, float] = {}

    for chunk_start in range(0, n_keys, CHUNK):
        chunk_keys = keys[chunk_start : chunk_start + CHUNK]  # (C,)
        # Outer product: (C, N)
        P = np.floor(
            np.outer(chunk_keys.astype(np.float64), nonces.astype(np.float64)) / q
        )  # (C, N)

        p_mean = P.mean(axis=1, keepdims=True)   # (C, 1)
        p_std = P.std(axis=1)                     # (C,)

        p_norm = P - p_mean                       # (C, N)

        # Dot product with t_norm gives numerator of Pearson r
        numerator = p_norm @ t_norm               # (C,)

        # Denominator
        n_samples = timings.shape[0]
        denominator = p_std * t_std * n_samples   # (C,)

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denominator == 0, 0.0, numerator / denominator)

        for i, k in enumerate(chunk_keys):
            correlations[int(k)] = float(corr[i])

        if verbose and (chunk_start // CHUNK) % 16 == 0:
            pct = min(100, (chunk_start + CHUNK) / n_keys * 100)
            print(f"    Progress: {pct:5.1f}%  ({chunk_start + CHUNK}/{n_keys})",
                  end="\r", flush=True)

    if verbose:
        print(" " * 60, end="\r")   # clear progress line

    return correlations


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="attack.py",
        description=(
            "DSS Timing Side-Channel Attack — demonstrates how non-constant-time\n"
            "modular arithmetic leaks secret key material through timing differences."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--key-bits",
        type=int,
        default=DEFAULT_KEY_BITS,
        metavar="BITS",
        help=f"Approximate bit-length of the key space (default: {DEFAULT_KEY_BITS}). "
             "Controls the group order q used.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        metavar="N",
        help=f"Number of timing samples to generate (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=NOISE_STD,
        metavar="STD",
        help=f"Standard deviation of Gaussian timing noise in ns (default: {NOISE_STD}).",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=DEFAULT_Q,
        metavar="Q",
        help=f"Override the group order prime q (default: {DEFAULT_Q}).",
    )
    parser.add_argument(
        "--save-dataset",
        type=str,
        default="dataset/sample_data.csv",
        metavar="PATH",
        help='Save generated dataset to this CSV path (default: "dataset/sample_data.csv").',
    )
    parser.add_argument(
        "--load-dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Load an existing dataset CSV instead of generating a new one.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Skip generating visualisation plots.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the timing attack demonstration."""
    args = parse_args()

    verbose = not args.quiet
    plot = not args.no_plot
    q = args.q

    # ------------------------------------------------------------------
    # Dataset acquisition
    # ------------------------------------------------------------------
    if args.load_dataset:
        df = load_dataset(args.load_dataset)
        # When loading from disk we don't know the true key
        secret_key: int | None = None
        if verbose:
            print("[!] True key unknown (loaded from file) — accuracy check skipped.")
    else:
        secret_key = generate_secret_key(args.key_bits, q)
        if verbose:
            print(f"[*] Generated secret key: {secret_key}  (keep this secret!)")
        df = generate_dataset(
            secret_key=secret_key,
            q=q,
            n_samples=args.samples,
            noise_std=args.noise,
        )

        # Save dataset
        if args.save_dataset:
            os.makedirs(os.path.dirname(args.save_dataset) or ".", exist_ok=True)
            save_dataset(df, args.save_dataset)

    # ------------------------------------------------------------------
    # Run the attack
    # ------------------------------------------------------------------
    recovered = run_attack(df, secret_key=secret_key, q=q, verbose=verbose, plot=plot)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("  Attack Complete")
        print("=" * 60)
        if secret_key is not None:
            if recovered == secret_key:
                print("  Result : KEY FULLY RECOVERED  ✓")
            else:
                print(f"  Result : Partial recovery — got {recovered}, expected {secret_key}")
        else:
            print(f"  Result : Recovered key candidate = {recovered}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
