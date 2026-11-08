"""
cpdr_baseline.py — Fair CPDR (Clifford Perturbation Data Regression) Baseline
══════════════════════════════════════════════════════════════════════════════
Implements the key competitor method from Zhang et al. [25] in a fair
head-to-head comparison against PDR-QPD.

Why this is a FAIR comparison (fixing the original paper's bug):
    ✓ Same FakeFez noise model as PDR-QPD (original used different backends)
    ✓ Same perturbation range δθ ∈ [−0.2, 0.2]
    ✓ Same shot count (SHOTS = 8192)
    ✓ Same calibration size K

Key difference from PDR-QPD:
    ✗ No QPD — the FULL 2-qubit circuit is measured directly
    ✗ Ridge regression (not TheilSen)
    ✗ No per-variant noise characterisation

This isolates the effect of QPD decomposition + per-variant calibration
as the distinctive contribution of PDR-QPD.
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate

from config import K_CAL_SCALAR, DELTA_MAX, SEED


class CPDRBaseline:
    """
    Clifford Perturbation Data Regression — fair baseline.

    One trial:
        1. Generate K perturbed versions of the full 2-qubit circuit.
        2. For each: measure ZZ (noisy) + compute ideal ZZ (Statevector).
        3. Fit Ridge regression: ideal = α · noisy + β.
        4. Correct the target circuit's noisy ZZ.

    Reported metrics:
        noisy_zz  — unmitigated ZZ from the full circuit
        cpdr_zz   — CPDR-corrected ZZ
        r2        — regression fit quality
        fit_slope — regression slope (for channel interpretation)
    """

    def __init__(self, K: int = K_CAL_SCALAR,
                 delta_max: float = DELTA_MAX,
                 seed: int = SEED):
        self.K         = K
        self.delta_max = delta_max
        self.rng       = np.random.default_rng(seed)
        self.model     = Ridge(alpha=0.1)

        # Reported after each trial
        self.fit_slope     = np.nan
        self.fit_intercept = np.nan
        self.r2            = np.nan

    def run_trial(self, target_qc: QuantumCircuit, executor) -> dict:
        """
        Run one CPDR trial.

        Parameters
        ----------
        target_qc : QuantumCircuit
            Full 2-qubit target circuit (RX-RX-CZ).
        executor : QuantumExecutor

        Returns
        -------
        dict:
            noisy_zz : float — unmitigated ZZ
            cpdr_zz  : float — CPDR-corrected ZZ
            r2       : float — regression R²
            fit_slope: float — regression slope
        """
        n_rx = sum(1 for i in target_qc.data if i.operation.name == 'rx')

        # ── Step 1: Generate K perturbed full circuits ────────────────────────
        cal_circuits = []
        for _ in range(self.K):
            delta   = self.rng.uniform(-self.delta_max, self.delta_max,
                                        size=max(n_rx, 1))
            cal_circuits.append(self._perturb(target_qc, delta))

        # ── Step 2: Ideal values (Statevector, free) ──────────────────────────
        ideal_zz = np.array([executor.get_ideal_zz(qc) for qc in cal_circuits])

        # ── Step 3: Noisy measurements (ONE batch) ────────────────────────────
        all_counts = executor.batch_run(cal_circuits)
        noisy_zz   = np.array([
            executor._counts_to_zz(c, target_qc.num_qubits)
            for c in all_counts])

        # ── Step 4: Fit Ridge regression (ideal = f(noisy)) ──────────────────
        X = noisy_zz.reshape(-1, 1)
        self.model.fit(X, ideal_zz)

        pred    = self.model.predict(X)
        ss_res  = np.sum((ideal_zz - pred) ** 2)
        ss_tot  = np.sum((ideal_zz - ideal_zz.mean()) ** 2)
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        slope_lin, intercept_lin, _, _, _ = stats.linregress(noisy_zz, ideal_zz)
        self.fit_slope     = float(self.model.coef_[0])
        self.fit_intercept = float(self.model.intercept_)

        # ── Step 5: Correct target circuit ───────────────────────────────────
        target_noisy_counts = executor.batch_run([target_qc])[0]
        target_noisy_zz     = executor._counts_to_zz(
            target_noisy_counts, target_qc.num_qubits)
        cpdr_zz = float(self.model.predict([[target_noisy_zz]])[0])

        return {
            'noisy_zz' : target_noisy_zz,
            'cpdr_zz'  : cpdr_zz,
            'r2'       : self.r2,
            'fit_slope': self.fit_slope,
        }

    def _perturb(self, qc: QuantumCircuit, deltas: np.ndarray) -> QuantumCircuit:
        """Apply angle perturbations δθ to all RX gates."""
        new_qc   = QuantumCircuit(qc.num_qubits)
        rx_count = 0
        for inst in qc.data:
            gate   = inst.operation
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            if gate.name == 'rx' and rx_count < len(deltas):
                new_qc.append(RXGate(gate.params[0] + deltas[rx_count]), qubits)
                rx_count += 1
            else:
                new_qc.append(gate, qubits)
        return new_qc
