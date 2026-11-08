"""
channel_model.py — Channel-Theoretic Framework for PDR-QPD
══════════════════════════════════════════════════════════
Connects the PDR regression parameters to formal quantum channel theory:

1. Kraus Representation → justifies linearity of PDR regression
2. Depolarizing Channel → interprets regression slope m as noise parameter ε
3. Stinespring Representation → explains slope > 1 (coherent over-rotation)
4. Gentle Measurement Lemma → bounds perturbation fidelity
5. Diagonal Pauli-Channel Bloch Model → robust per-axis noise characterisation
"""
import numpy as np
from scipy import stats
from typing import Tuple, Dict


# ── 1. Kraus Representation ──────────────────────────────────────────────────

def kraus_operators_depolarizing(epsilon: float) -> list:
    """
    Kraus operators for the single-qubit depolarizing channel:
        Φ_ε(ρ) = (1−ε)ρ + ε·(I/2)
        K₀ = √(1−ε)·I,  K₁ = √(ε/3)·X,  K₂ = √(ε/3)·Y,  K₃ = √(ε/3)·Z

    Linearity Justification:
        For small ε, K₀ ≈ I dominates → first-order expansion gives
        a linear map on expectation values: <O>_noisy ≈ (1−ε)<O>_ideal.
        This is the formal justification for the PDR linear regression assumption.
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    return [
        np.sqrt(1 - epsilon) * I,
        np.sqrt(epsilon / 3) * X,
        np.sqrt(epsilon / 3) * Y,
        np.sqrt(epsilon / 3) * Z,
    ]


def verify_kraus_completeness(kraus_ops: list) -> bool:
    """Check Σᵢ Kᵢ†Kᵢ = I (completeness relation)."""
    total = sum(K.conj().T @ K for K in kraus_ops)
    return bool(np.allclose(total, np.eye(2), atol=1e-10))


# ── 2. Regression slope interpretation ───────────────────────────────────────

def interpret_regression_params(fit_slope: float,
                                 fit_intercept: float) -> Dict:
    """
    Channel-theoretic interpretation of PDR regression parameters.

    Regression direction: ideal = fit_slope · noisy + intercept
    For depolarizing noise: noisy = (1−ε)·ideal → fit_slope = 1/(1−ε) > 1
    Damping m = 1/fit_slope ≈ 1−ε  (paper's Table 4.1 convention, m < 1)

    Stinespring interpretation:
        fit_slope > 1 → depolarising (signal damping)
        fit_slope < 1 → coherent over-rotation (amplification error)
    """
    damping_m = 1.0 / fit_slope if abs(fit_slope) > 1e-6 else np.nan
    epsilon   = 1.0 - damping_m if not np.isnan(damping_m) else np.nan

    if not np.isnan(damping_m) and damping_m < 1.0:
        channel_type = "depolarizing"
        desc = f"Depolarizing: ε={abs(epsilon):.4f}, m={damping_m:.4f}"
    elif not np.isnan(damping_m) and damping_m > 1.0:
        channel_type = "coherent_over_rotation"
        desc = f"Coherent over-rotation: m={damping_m:.4f}"
    else:
        channel_type = "noiseless"
        desc = "Near-noiseless: m ≈ 1"

    return {
        "fit_slope"    : float(fit_slope),
        "fit_intercept": float(fit_intercept),
        "damping_m"    : float(damping_m) if not np.isnan(damping_m) else 1.0,
        "epsilon"      : float(abs(epsilon)) if not np.isnan(epsilon) else 0.0,
        "channel_type" : channel_type,
        "description"  : desc,
    }


# ── 3. Gentle Measurement Lemma ──────────────────────────────────────────────

def gentle_fidelity_bound(delta_theta: float) -> float:
    """
    Lower bound on fidelity between original and perturbed circuit states.
    From Gentle Measurement Lemma: F ≥ cos²(|δθ|/2).
    For δθ ∈ [−0.2, 0.2]: bound ≥ cos²(0.1) ≈ 0.9900.
    """
    return float(np.cos(abs(delta_theta) / 2) ** 2)


# ── 4. Diagonal Pauli-Channel Bloch Model ────────────────────────────────────

def fit_bloch_matrix(ideal_bloch: np.ndarray,
                     noisy_bloch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the diagonal Pauli-channel noise model per Bloch axis.

    Physical justification:
        NISQ noise is well-approximated by Pauli channels, which have a
        diagonal Bloch map: T = diag(T_X, T_Y, T_Z).
        This avoids the multicollinearity problem of full 3×3 lstsq fitting
        when one Bloch axis (X for pure-RX circuits) has near-zero variance.

    Model: noisy_a = T_aa · ideal_a + t_a  for each axis a ∈ {X, Y, Z}

    For depolarizing: T_XX = T_YY = T_ZZ = 1−ε,  t = 0.

    Parameters
    ----------
    ideal_bloch : np.ndarray (K, 3) — ideal [X, Y, Z] per calibration circuit
    noisy_bloch : np.ndarray (K, 3) — noisy [X, Y, Z] per calibration circuit

    Returns
    -------
    T : np.ndarray (3, 3) diagonal — Bloch shrinkage matrix
    t : np.ndarray (3,)   — per-axis bias
    """
    T = np.zeros((3, 3))
    t = np.zeros(3)

    for a in range(3):
        x_vals = ideal_bloch[:, a]
        y_vals = noisy_bloch[:, a]

        # If this axis has very low variance (e.g., X ≈ 0 for RX circuits),
        # use a robust estimate instead of linregress (which diverges).
        variance = np.var(x_vals)
        if variance < 1e-4:
            # Estimate as ratio of means (robust for near-zero axis)
            T[a, a] = (np.mean(y_vals) / np.mean(x_vals)
                       if abs(np.mean(x_vals)) > 1e-6 else 0.97)
            t[a] = np.mean(y_vals) - T[a, a] * np.mean(x_vals)
        else:
            slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
            T[a, a] = float(slope)
            t[a]    = float(intercept)

    return T, t


def invert_channel_from_bloch(r_noisy: np.ndarray,
                               T: np.ndarray,
                               t: np.ndarray) -> np.ndarray:
    """
    Invert diagonal Pauli-channel: r_ideal_est = T⁻¹ · (r_noisy − t).
    For diagonal T: T⁻¹ = diag(1/T_XX, 1/T_YY, 1/T_ZZ).
    Clips T diagonal to avoid division by near-zero.
    """
    T_diag   = np.diag(T).copy()
    T_diag   = np.where(np.abs(T_diag) > 0.05, T_diag, 0.97)  # fallback
    T_inv    = np.diag(1.0 / T_diag)
    return T_inv @ (r_noisy - t)
