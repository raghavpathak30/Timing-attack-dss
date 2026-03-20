"""
visualize.py — Visualisation utilities for the DSS timing side-channel attack.

Produces two plots:
1. **Timing distribution histogram** — shows that the measured timings follow
   a distribution that varies with the secret key (the leakage signal).
2. **Correlation vs. key-guess scatter plot** — shows how Pearson correlation
   between predicted timing cost and observed timing peaks at the true key.

Both plots are saved as PNG files in the ``plots/`` directory.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    from scipy.stats import gaussian_kde as _gaussian_kde
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Plot 1 — Timing distribution
# ---------------------------------------------------------------------------

def plot_timing_histogram(
    df: pd.DataFrame,
    output_path: str = "plots/timing_histogram.png",
    title: str = "Timing Distribution of Vulnerable Modular Multiplication",
) -> None:
    """Plot a histogram of the observed timing measurements.

    A non-constant-time implementation produces a *bimodal* or *skewed*
    distribution because different input ranges require a different number of
    reduction steps.  This contrasts with a constant-time implementation which
    would produce a near-uniform spike.

    Parameters
    ----------
    df:
        Dataset DataFrame with a ``timing_ns`` column.
    output_path:
        Destination PNG file path.
    title:
        Plot title shown on the figure.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    timings = df["timing_ns"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        timings,
        bins=80,
        color="#2196F3",
        edgecolor="#0d47a1",
        alpha=0.85,
        density=True,
    )

    # Overlay a KDE for smoothness (requires scipy; skipped gracefully if absent)
    if _SCIPY_AVAILABLE:
        kde = _gaussian_kde(timings, bw_method="scott")
        xs = np.linspace(timings.min(), timings.max(), 500)
        ax.plot(xs, kde(xs), color="#e53935", linewidth=2, label="KDE")
        ax.legend(fontsize=11)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Timing (nanoseconds)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} ns"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[+] Timing histogram saved → {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Correlation vs. key-guess
# ---------------------------------------------------------------------------

def plot_correlation_vs_guesses(
    correlations: dict[int, float],
    true_key: int | None = None,
    output_path: str = "plots/correlation_vs_guesses.png",
    title: str = "Pearson Correlation vs. Key Guess",
) -> None:
    """Plot how the Pearson correlation coefficient varies across key guesses.

    The true key produces the *highest* absolute correlation, forming a clear
    peak in the plot.  This visualises why the attack succeeds.

    Parameters
    ----------
    correlations:
        Mapping from key-guess (int) to Pearson correlation coefficient.
    true_key:
        If provided, the true key is highlighted with a vertical dashed line.
    output_path:
        Destination PNG file path.
    title:
        Plot title.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    guesses = np.array(sorted(correlations.keys()))
    corrs = np.array([correlations[g] for g in guesses])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(guesses, corrs, color="#4CAF50", linewidth=1.2, alpha=0.85,
            label="Correlation")
    ax.scatter(guesses, corrs, s=4, color="#4CAF50", alpha=0.6)

    if true_key is not None:
        ax.axvline(
            true_key,
            color="#e53935",
            linewidth=2,
            linestyle="--",
            label=f"True key = {true_key}",
        )
        # Annotate peak
        ax.annotate(
            f"True key\n{true_key}",
            xy=(true_key, correlations.get(true_key, 0)),
            xytext=(true_key + len(guesses) * 0.03, max(corrs) * 0.85),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10,
            color="#b71c1c",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Key Guess", fontsize=12)
    ax.set_ylabel("Pearson Correlation", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[+] Correlation plot saved → {output_path}")
