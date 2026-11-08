"""
config.py — Global constants for PDR-QPD v2
All parameters in one place for easy tuning.
"""
import numpy as np
import os

# ── Target circuit angles ────────────────────────────────────────────────────
THETA_0 = 0.6          # RX angle on qubit 0
THETA_1 = 0.8          # RX angle on qubit 1
CUT_PAIR = (0, 1)      # QPD cut between qubit 0 and qubit 1

# Analytically derived ideal: <ZZ> = cos(θ₀)·cos(θ₁) for RX-RX-CZ circuit
# CZ is diagonal in Z-basis → transparent to ZZ measurement.
# True ideal value (verified by Statevector in executor):
IDEAL_ZZ = float(np.cos(THETA_0) * np.cos(THETA_1))   # ≈ 0.57502

# ── Experiment parameters ────────────────────────────────────────────────────
N_TRIALS      = 100    # Statistical trials for validation
SHOTS         = 8192   # Unified shot count — same for calibration AND target
K_CAL_SCALAR  = 20     # Calibration circuits for Linear / GPR modes
K_CAL_BLOCH   = 15     # Calibration circuits for Bloch-matrix mode (3× heavier)
DELTA_MAX     = 0.2    # Perturbation range δθ ~ U(-0.2, 0.2)
SEED          = 42     # Global random seed

# ── Gentle Measurement Lemma bound ──────────────────────────────────────────
# For δθ ∈ [-0.2, 0.2]: fidelity ≥ cos²(0.1) ≈ 0.9950
# Ensures perturbed circuits stay in the local noise manifold.
FIDELITY_BOUND = float(np.cos(DELTA_MAX / 2) ** 2)    # ≈ 0.9950

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
