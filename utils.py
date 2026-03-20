"""
utils.py — Core simulation utilities for the DSS timing side-channel attack.

This module simulates a vulnerable cryptographic operation (x * r mod q) that
leaks timing information because the modular reduction is *not* constant-time.
An attacker can observe these timings and—using statistical correlation—recover
the secret scalar x bit-by-bit.

Cryptographic background
------------------------
In DSS (Digital Signature Standard) and similar schemes the signer computes
  s = k^{-1} * (H(m) + x * r) mod q
where x is the long-term private key.  If the implementation uses a naïve
loop-based modular reduction the number of iterations (and therefore the wall-
clock time) depends on the intermediate value, leaking information about x.

We distil the vulnerability to the core operation:
  result = (x * r) mod q
and show how an attacker who can measure execution time can recover x.
"""

from __future__ import annotations

import secrets
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cryptographic parameters
# ---------------------------------------------------------------------------

# A 16-bit prime used as the DSS-like group order q.
# (Real DSS uses a 256-bit prime; 16-bits lets the demo run quickly.)
DEFAULT_Q: int = 65521
DEFAULT_KEY_BITS: int = 16
DEFAULT_N_SAMPLES: int = 5000
NOISE_STD: float = 5.0       # standard deviation of Gaussian noise (nanoseconds)
BASE_TIMING_NS: float = 200.0   # constant overhead per operation (ns)
COST_PER_ITER_NS: float = 10.0  # additional ns per reduction iteration (the leak)


# ---------------------------------------------------------------------------
# Vulnerable operation
# ---------------------------------------------------------------------------

def vulnerable_mod_mult(
    x: int,
    r: int,
    q: int,
    base_ns: float = BASE_TIMING_NS,
    cost_per_iter_ns: float = COST_PER_ITER_NS,
) -> tuple[int, int, float]:
    """Simulate a non-constant-time modular multiplication.

    The deliberate vulnerability: reduction is modelled as a subtraction loop
    whose iteration count — and therefore execution time — depends on the input
    ``x * r``.  The iteration count is::

        num_iters = floor(x * r / q)

    This is the *leaking quantity*: it correlates with ``x`` and is observable
    (approximately) through timing measurements.

    To make the simulation reproducible and independent of OS scheduler noise,
    the timing is *synthesised* from the iteration count rather than measured
    with ``time.perf_counter_ns()``:

        timing_ns = base_ns + num_iters * cost_per_iter_ns

    Gaussian noise is added by the caller (see :func:`add_gaussian_noise`).

    Parameters
    ----------
    x:
        Secret scalar (private key or its candidate).
    r:
        Public nonce / per-signature value chosen uniformly at random in [1, q).
    q:
        The group order (prime).
    base_ns:
        Constant baseline cost of the operation in nanoseconds.
    cost_per_iter_ns:
        Additional nanoseconds charged per reduction iteration.

    Returns
    -------
    result:
        ``(x * r) mod q``
    num_iters:
        Number of reduction iterations (the side-channel signal).
    raw_timing_ns:
        Synthetic timing before noise: ``base_ns + num_iters * cost_per_iter_ns``.
    """
    product: int = x * r

    # Number of times we would subtract q in a naïve loop (the leaking quantity)
    num_iters: int = product // q
    result: int = product - num_iters * q  # equivalent to product % q

    # Synthetic timing derived from the iteration count (the vulnerability)
    raw_timing_ns: float = base_ns + num_iters * cost_per_iter_ns

    return result, num_iters, raw_timing_ns


def add_gaussian_noise(timing: float, std: float = NOISE_STD) -> float:
    """Add realistic Gaussian noise to a timing measurement.

    Real hardware measurements are affected by OS scheduling, cache effects,
    branch-predictor state, and other sources of noise that are well-modelled
    by a Gaussian distribution.

    Parameters
    ----------
    timing:
        The raw measured time in nanoseconds.
    std:
        Standard deviation of the noise distribution (nanoseconds).

    Returns
    -------
    float
        ``timing`` perturbed by zero-mean Gaussian noise with the given std.
    """
    noise: float = np.random.normal(loc=0.0, scale=std)
    return max(0.0, timing + noise)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    secret_key: int,
    q: int = DEFAULT_Q,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_std: float = NOISE_STD,
) -> pd.DataFrame:
    """Generate a dataset of timing measurements for the vulnerable operation.

    For each sample a random nonce ``r`` is drawn and the timing of
    ``secret_key * r mod q`` is recorded (with added noise).  The dataset
    contains both public information (``r``, noisy timing) and the ground-truth
    result for verification purposes.

    Parameters
    ----------
    secret_key:
        The 16-bit private key ``x`` to simulate.
    q:
        Group order (prime).
    n_samples:
        Number of (r, timing) pairs to generate.
    noise_std:
        Standard deviation of Gaussian noise added to each measurement.

    Returns
    -------
    pd.DataFrame
        Columns: ``r``, ``timing_ns``, ``result``
        - ``r``          : random nonce in [1, q)
        - ``timing_ns``  : noisy timing measurement in nanoseconds
        - ``result``     : true value of (secret_key * r) mod q
    """
    rng = np.random.default_rng(seed=42)
    nonces: np.ndarray = rng.integers(low=1, high=q, size=n_samples)

    timings: list[float] = []
    results: list[int] = []
    iter_counts: list[int] = []

    for r_val in nonces:
        result, num_iters, raw_timing = vulnerable_mod_mult(int(secret_key), int(r_val), q)
        noisy_timing = add_gaussian_noise(raw_timing, std=noise_std)
        timings.append(noisy_timing)
        results.append(result)
        iter_counts.append(num_iters)

    df = pd.DataFrame(
        {
            "r": nonces,
            "timing_ns": timings,
            "result": results,
            "num_iters": iter_counts,
        }
    )
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Persist the dataset to a CSV file.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`generate_dataset`.
    path:
        Destination file path (e.g. ``"dataset/sample_data.csv"``).
    """
    df.to_csv(path, index=False)
    print(f"[+] Dataset saved → {path}  ({len(df)} samples)")


def load_dataset(path: str) -> pd.DataFrame:
    """Load a previously saved dataset from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file created by :func:`save_dataset`.

    Returns
    -------
    pd.DataFrame
        Reconstructed dataset with columns ``r``, ``timing_ns``, ``result``.
    """
    df = pd.read_csv(path)
    print(f"[+] Dataset loaded ← {path}  ({len(df)} samples)")
    return df


# ---------------------------------------------------------------------------
# Key generation helpers
# ---------------------------------------------------------------------------

def generate_secret_key(key_bits: int = DEFAULT_KEY_BITS, q: int = DEFAULT_Q) -> int:
    """Generate a random secret key in the range [1, q).

    Parameters
    ----------
    key_bits:
        Approximate bit-length of the key space (informational only; the key is
        drawn uniformly from [1, q)).
    q:
        Group order; the key is bounded by ``q``.

    Returns
    -------
    int
        Secret key ``x`` with ``1 <= x < q``.
    """
    return secrets.randbelow(q - 1) + 1


# ---------------------------------------------------------------------------
# Timing cost predictor (used in the attack)
# ---------------------------------------------------------------------------

def predicted_timing_cost(x_guess: int, r: int, q: int) -> int:
    """Compute the predicted number of reduction iterations for a key guess.

    This is the *attacker's model* of how many loop iterations the vulnerable
    reduction would take for the pair (``x_guess``, ``r``).  A higher value
    means more loop iterations, hence longer timing.

    The model is: number of times q is subtracted from x_guess * r, i.e.
        floor((x_guess * r) / q)

    Parameters
    ----------
    x_guess:
        Candidate key value being tested.
    r:
        Nonce from the dataset.
    q:
        Group order.

    Returns
    -------
    int
        Predicted iteration count ≈ floor(x_guess * r / q).
    """
    return (x_guess * int(r)) // q
