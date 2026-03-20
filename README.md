# Timing-attack-dss

A Python-based simulation of a timing side-channel attack on DSS-style modular arithmetic. Demonstrates how non-constant-time implementations leak secret keys using statistical correlation analysis under noisy conditions.

---

## Table of Contents

- [Overview](#overview)
- [Vulnerability Explained](#vulnerability-explained)
- [Attack Steps](#attack-steps)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Example Output](#example-output)
- [Real-World Implications](#real-world-implications)
- [References](#references)

---

## Overview

The **Digital Signature Standard (DSS)** — and similar discrete-logarithm-based schemes — requires the signer to compute:

```
s = k⁻¹ · (H(m) + x · r)  mod q
```

where `x` is the long-term **private key**.  If the underlying modular arithmetic is implemented naïvely (e.g., using a subtraction loop rather than a constant-time reduction), the number of loop iterations — and hence execution time — leaks information about `x`.

This project distils that vulnerability to the core operation:

```
result = (x · r) mod q
```

and shows, step by step, how an attacker who can collect timing measurements can **fully recover the secret key** using Pearson correlation analysis.

---

## Vulnerability Explained

### The leaky operation

A naïve modular reduction looks like:

```python
value = x * r
while value >= q:       # each iteration takes measurable time
    value -= q
```

The **number of iterations** is exactly `floor(x · r / q)`.  This quantity is correlated with `x`: for larger `x` the product `x · r` is larger, so more subtractions are needed and the operation takes longer.

### Why this matters

An attacker who can measure execution times (e.g., via network latency, cache state, power consumption, or direct timing) can use those measurements to reconstruct the secret key, even in the presence of realistic noise.

### Constant-time alternative

A secure implementation uses the built-in `%` operator (or a branchless Montgomery reduction), which runs in constant time regardless of the input, removing the leakage channel.

---

## Attack Steps

The attack proceeds in five steps:

1. **Collect timing measurements.**  Run the vulnerable operation for many random nonces `r ∈ [1, q)` and record the execution time for each.  Add realistic Gaussian noise to model hardware jitter.

2. **Build the prediction model.**  For each key candidate `x_guess`, compute the predicted iteration count:
   ```
   predicted_cost(x_guess, r) = floor(x_guess · r / q)
   ```

3. **Compute Pearson correlation.**  For each `x_guess`, calculate the Pearson correlation coefficient between the predicted-cost vector and the observed-timing vector across all `N` samples.

4. **Select the best candidate.**  The `x_guess` that maximises the absolute correlation is the recovered key.  When the guess is correct the prediction perfectly models the leaking computation, yielding a correlation near 1.0.

5. **Report and visualise.**  Print the recovered key, check against the ground truth, and plot the timing distribution and correlation landscape.

---

## Project Structure

```
timing-attack-dss/
│
├── attack.py          # Main attack script (CLI entry point)
├── utils.py           # Cryptographic simulation & dataset utilities
├── visualize.py       # Histogram and correlation plots
│
├── dataset/
│   └── sample_data.csv    # Generated timing dataset (5 000 samples)
│
├── plots/
│   ├── timing_histogram.png        # Timing distribution plot
│   └── correlation_vs_guesses.png  # Correlation landscape plot
│
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.9 or later

### Installation

```bash
# Clone the repository
git clone https://github.com/raghavpathak30/Timing-attack-dss.git
cd Timing-attack-dss

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run with default settings

```bash
python attack.py
```

Generates 5 000 timing samples, runs the full correlation sweep across all 65 520 key candidates, prints the recovered key, and saves two plots.

### Command-line options

```
usage: attack.py [-h] [--key-bits BITS] [--samples N] [--noise STD]
                 [--q Q] [--save-dataset PATH] [--load-dataset PATH]
                 [--no-plot] [--quiet] [--normalize-timing]

options:
  --key-bits BITS        Approximate key-space bit-length (default: 16)
  --samples N            Number of timing samples to generate (default: 5000)
  --noise STD            Gaussian noise std-dev in ns (default: 5.0)
  --q Q                  Override the group order prime q (default: 65521)
  --save-dataset PATH    Path to save the generated dataset CSV
  --load-dataset PATH    Load an existing dataset instead of generating
  --no-plot              Skip generating visualisation plots
  --quiet                Suppress progress output
  --normalize-timing     Normalize timing values before correlation
```

### Examples

```bash
# Larger dataset for a harder noise environment
python attack.py --samples 10000 --noise 20.0

# Use a pre-generated dataset
python attack.py --load-dataset dataset/sample_data.csv

# Minimal output, no plots
python attack.py --no-plot --quiet
```

---

## Example Output

```
[*] Generated secret key: 35645  (keep this secret!)
[+] Dataset saved → dataset/sample_data.csv  (5000 samples)

============================================================
  DSS Timing Side-Channel Attack
============================================================
  Group order q      : 65521
  Dataset size       : 5000 samples
  True secret key    : 35645
============================================================

[+] Timing histogram saved → plots/timing_histogram.png
[*] Testing 65520 key candidates …
[+] Recovered key  : 35645
[+] Actual key     : 35645
[+] Status         : SUCCESS
[+] Best |corr|    : 1.000000
[+] Correlation plot saved → plots/correlation_vs_guesses.png

============================================================
  Attack Complete
============================================================
  Actual key    : 35645
  Recovered key : 35645
  Status        : SUCCESS
============================================================
```

### Generated plots

| Timing distribution | Correlation landscape |
|---|---|
| `plots/timing_histogram.png` | `plots/correlation_vs_guesses.png` |

The histogram shows the non-uniform (bimodal / skewed) timing distribution that results from the variable-length reduction loop. The correlation plot shows a sharp peak exactly at the actual key value.

---

## Real-World Implications

### Historical exploits

| CVE / Attack | System | Year |
|---|---|---|
| [Kocher timing attack](https://www.paulkocher.com/doc/TimingAttacks.pdf) | RSA / DH | 1996 |
| OpenSSL RSA timing (CVE-2003-0147) | OpenSSL | 2003 |
| Lucky Thirteen (CVE-2013-0169) | TLS CBC-MAC | 2013 |
| [Minerva](https://minerva.crocs.fi.muni.cz/) | ECDSA (smartcards) | 2019 |
| [TPM-FAIL](https://tpm.fail/) | TPM 2.0 ECDSA | 2019 |

### Why this matters today

1. **Key recovery from remote measurements** — even network-level timing (with statistical averaging) can be sufficient to break non-constant-time implementations.
2. **Side-channel resilience is hard** — compilers, CPUs, and caches can re-introduce non-constant-time behaviour even in "fixed" code.
3. **Formal verification** — tools such as ct-verif and Jasmin are now used to *prove* constant-time behaviour at the assembly level.

### Mitigations

- Use constant-time arithmetic libraries (e.g., `libsodium`, `BoringSSL`).
- Avoid data-dependent branches and memory accesses in cryptographic code.
- Add dummy operations or random delays as a last resort (blinding).
- Audit cryptographic code with static analysis tools.

---

## References

- Kocher, P. (1996). *Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS, and Other Systems.*
- Brumley, D. & Boneh, D. (2005). *Remote Timing Attacks Are Practical.*
- FIPS 186-5. *Digital Signature Standard (DSS)*. NIST, 2023.
- Bernstein, D. J. (2005). *Cache-timing attacks on AES.*

