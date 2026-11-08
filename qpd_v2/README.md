# Adaptive QPD-PDR v3 — Complete Framework

## Paper Contribution Summary

**Core Claim (Theorem 1):**  
QPD+PDR outperforms Direct+PDR if and only if:

    γ · ε_fragment(O) < ε_full_circuit(O)
    
    where ε_full_circuit(O) = ε_base(O) + n_SWAP · ε_SWAP

**Key Results (Section III):**
- Decision is independent of β (mitigation factor) — hardware-only
- CZ+ZZ regime: Direct wins at n_SWAP=0, QPD wins at n_SWAP≥1
- CNOT+ZZ or CZ+XX regime: QPD wins even at n_SWAP=0

## File Structure

```
qpd_v2/
├── requirements.txt          Pinned dependencies
├── config.py                 Global constants (edit here only)
├── qpd_lib.py                QPD decomposition (FIXED circuit cutting)
├── channel_model.py          Kraus/Bloch/depolarising theory
├── executor.py               SamplerV2 unified execution + batching
├── calibration.py            Structure-aware perturbation calibrator
├── pdr_regression.py         Linear / GPR / Bloch regression modes
├── cpdr_baseline.py          Fair CPDR comparison (same noise backend)
├── metrics.py                Fidelity, CI, t-test, epsilon, summary
├── adaptive_framework.py     ← NEW: Theorem 1 decision engine
├── main_simulation.py        100-trial comparison (5 methods)
└── main_adaptive.py          ← NEW: 3 experiments validating Theorem 1
```

## How to Run

### Experiment A — 100-Trial Method Comparison (original)
```bash
python main_simulation.py
```
Output: `results/fig1-4_*.png`, `results/trial_data.csv`

### Experiment B — Theorem 1 Validation (new, adaptive)
```bash
python main_adaptive.py
```
Output: `results/adaptive_exp1_breakeven.png`
        `results/adaptive_exp2_decision.png`  
        `results/adaptive_exp3_scaling.png`
        `results/adaptive_summary.csv`

## Quick Adaptive Framework Usage

```python
from executor import QuantumExecutor
from adaptive_framework import AdaptiveQPDPDR
from qpd_lib import build_target_circuit

executor  = QuantumExecutor(shots=8192)
framework = AdaptiveQPDPDR(executor, executor.backend, observable='ZZ')
target    = build_target_circuit(0.6, 0.8)

result = framework.run(target, cut_qubit_pair=(0, 1))
print(result['decision'])        # 'qpd' or 'direct'
print(result['mitigated_value']) # corrected expectation value
print(result['report'])          # full Theorem 1 audit trail
```

## Fixes from Original Thesis Code

| Bug | Fix |
|-----|-----|
| QPD appended Z gates (never cut) | Rebuild circuit without CZ, insert Paulis |
| Ising RZZ circuit (wrong ideal) | Unified RX-RX-CZ circuit, ideal=cos(θ₀)cos(θ₁) |
| CPDR used different noise backend | Same QuantumExecutor for all methods |
| Shot asymmetry | SHOTS=8192 everywhere |
| No requirements.txt | Pinned versions added |

## Novel Contributions vs Literature

| Prior Work | What it does | Gap |
|---|---|---|
| HIC (Pal et al. 2026) | WHERE to cut (noise-aware location) | Doesn't decide WHETHER to cut; no PDR |
| Majumdar & Wood (2022) | Wire cutting + DEVT tomographic mitigation | Gate cutting/QPD not used; no adaptive decision |
| **This work** | WHETHER to cut via Theorem 1 + PDR per fragment | First adaptive gate-cut/no-cut decision with regression mitigation |
