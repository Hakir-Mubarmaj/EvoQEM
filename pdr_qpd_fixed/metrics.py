"""
metrics.py — Evaluation Metrics for PDR-QPD
════════════════════════════════════════════
Metrics beyond the paper's original absolute error:

1. Absolute error |E_mitigated − E_ideal|          (original)
2. Fraction of trials improved over noisy baseline
3. 95% confidence interval (check if ideal is contained)
4. Paired t-test p-value
5. Depolarising ε = 1 − 1/fit_slope               (channel-theoretic, NEW)
6. Fidelity F = 1 − |E_mitigated − E_ideal|        (proxy, NEW)
   Note: True state fidelity requires full tomography; this is a
   single-observable proxy useful for ranking methods.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict


# ── Confidence Interval ───────────────────────────────────────────────────────

def compute_ci(samples: np.ndarray,
               confidence: float = 0.95) -> Tuple[float, float]:
    """
    Student-t confidence interval for the mean.

    Returns
    -------
    (lower, upper) bounds.
    """
    n    = len(samples)
    mean = np.mean(samples)
    sem  = stats.sem(samples)
    h    = sem * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean - h), float(mean + h)


# ── Hypothesis Test ───────────────────────────────────────────────────────────

def paired_ttest(errors_noisy: np.ndarray,
                  errors_mitigated: np.ndarray) -> Dict:
    """
    Paired t-test: H₀ = no difference in absolute errors.
    A significant p-value (p ≪ 0.05) confirms the improvement is real.
    """
    t_stat, p_val = stats.ttest_rel(errors_noisy, errors_mitigated)
    return {'t_statistic': float(t_stat), 'p_value': float(p_val)}


# ── Fraction Improved ─────────────────────────────────────────────────────────

def fraction_improved(errors_noisy: np.ndarray,
                       errors_mitigated: np.ndarray) -> float:
    """Fraction of trials where mitigation improved over noisy baseline."""
    return float(np.mean(errors_mitigated < errors_noisy))


# ── Channel-theoretic metrics ─────────────────────────────────────────────────

def compute_epsilon_from_slopes(fit_slopes: np.ndarray) -> np.ndarray:
    """
    Depolarising parameter ε = 1 − 1/fit_slope (per trial).

    Interpretation:
        fit_slope = 1/(1−ε) > 1 for depolarising noise
        → ε = 1 − 1/fit_slope ≈ noise strength

    For coherent over-rotation (fit_slope < 1):
        ε is negative → not a depolarising regime.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = 1.0 - 1.0 / np.where(fit_slopes != 0, fit_slopes, np.nan)
    return eps


def compute_damping_m(fit_slopes: np.ndarray) -> np.ndarray:
    """
    Damping parameter m = 1/fit_slope (paper Table 4.1 convention).
    m < 1 for depolarising, m > 1 for coherent over-rotation.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1.0 / np.where(fit_slopes != 0, fit_slopes, np.nan)


# ── Observable-proxy Fidelity ─────────────────────────────────────────────────

def observable_fidelity(mitigated_val: float, ideal_val: float) -> float:
    """
    Single-observable proxy fidelity: F = 1 − |E_mitigated − E_ideal|.

    Note: This is NOT the quantum state fidelity F(ρ, σ).
    It is a convenient [0,1]-normalised error proxy for ranking methods.
    Full state fidelity requires tomographic measurements.

    Returns
    -------
    float ∈ [0, 1], higher is better.
    """
    return float(max(0.0, 1.0 - abs(mitigated_val - ideal_val)))


# ── Summary aggregator ────────────────────────────────────────────────────────

def summarize(mitigated_vals: np.ndarray,
               noisy_vals: np.ndarray,
               fit_slopes: np.ndarray,
               ideal: float,
               method_name: str = "") -> Dict:
    """
    Compute all evaluation metrics for one method over N trials.

    Parameters
    ----------
    mitigated_vals : np.ndarray (N,)  — corrected expectation values
    noisy_vals     : np.ndarray (N,)  — unmitigated (standard QPD) values
    fit_slopes     : np.ndarray (N,)  — regression slopes per trial
    ideal          : float            — true noiseless expectation value
    method_name    : str

    Returns
    -------
    dict of all metrics (suitable for printing and CSV export).
    """
    mit_err   = np.abs(mitigated_vals - ideal)
    noisy_err = np.abs(noisy_vals     - ideal)
    ci_lo, ci_hi = compute_ci(mitigated_vals)
    ttest        = paired_ttest(noisy_err, mit_err)
    epsilons     = compute_epsilon_from_slopes(fit_slopes)
    damping_ms   = compute_damping_m(fit_slopes)
    obs_fid      = [observable_fidelity(v, ideal) for v in mitigated_vals]

    return {
        'method'              : method_name,
        'noisy_mean'          : float(np.mean(noisy_vals)),
        'noisy_abs_error'     : float(np.mean(noisy_err)),
        'mitigated_mean'      : float(np.mean(mitigated_vals)),
        'mitigated_std'       : float(np.std(mitigated_vals)),
        'mitigated_abs_error' : float(np.mean(mit_err)),
        'net_error_reduction' : float(np.mean(noisy_err) - np.mean(mit_err)),
        'fraction_improved'   : fraction_improved(noisy_err, mit_err),
        'ci_lower'            : ci_lo,
        'ci_upper'            : ci_hi,
        'ci_contains_ideal'   : bool(ci_lo <= ideal <= ci_hi),
        't_statistic'         : ttest['t_statistic'],
        'p_value'             : ttest['p_value'],
        'mean_epsilon'        : float(np.nanmean(epsilons)),
        'mean_damping_m'      : float(np.nanmean(damping_ms)),
        'mean_obs_fidelity'   : float(np.mean(obs_fid)),
        'ideal'               : ideal,
    }


def print_summary_table(all_summaries: list):
    """Print a formatted comparison table to stdout."""
    cols = [
        ('method',              'Method',        '<20s'),
        ('noisy_abs_error',     'Noisy Err',     '>10.5f'),
        ('mitigated_abs_error', 'Mit. Err',      '>10.5f'),
        ('fraction_improved',   'Frac. Imp.',    '>10.2%'),
        ('ci_contains_ideal',   'CI Has Ideal',  '>12s'),
        ('p_value',             'p-value',       '>12.2e'),
        ('mean_epsilon',        'ε (mean)',       '>10.4f'),
        ('mean_obs_fidelity',   'Obs. Fid.',     '>10.4f'),
    ]
    header = '  '.join(f'{label!s:>20}' for _, label, fmt in cols)
    print('\n' + '═' * len(header))
    print('PDR-QPD v2 — 100-Trial Comparison Summary')
    print('═' * len(header))
    print(header)
    print('─' * len(header))
    for s in all_summaries:
        row = ''
        for key, label, fmt in cols:
            val = s[key]
            if isinstance(val, bool):
                val = str(val)
            try:
                row += f'  {val:{fmt[1:]}}'
            except (TypeError, ValueError):
                row += f'  {str(val):<20}'
        print(row)
    print('═' * len(header) + '\n')
