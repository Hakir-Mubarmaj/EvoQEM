# replace the old generate_calibration_circuits in src/calibration.py with this version

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter, ParameterExpression, CircuitInstruction
from qiskit import QuantumCircuit

# Reproducibility seed (optional)
np.random.seed(42)

def _strip_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc with all 'measure' instructions removed."""
    qc_copy = qc.copy()
    qc_copy.data = [instr for instr in qc_copy.data if getattr(instr.operation, "name", "") != "measure"]
    # remove any trailing classical registers if present (safe to ignore as measurement removed)
    # Note: classical registers will remain but are harmless if no measure uses them.
    return qc_copy

def _ensure_measured(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc that has measurement on all qubits (safe if already measured)."""
    qc_meas = qc.copy()
    # if there is any 'measure' already, keep as-is; otherwise add measure_all
    has_measure = any(getattr(instr.operation, "name", "") == "measure" for instr in qc_meas.data)
    if not has_measure:
        qc_meas.measure_all()
    return qc_meas

def generate_calibration_circuits(base_subcircuits, num_samples=20):
    """
    Smart Calibration: Modifies EXISTING angles instead of adding new gates.
    Returns:
        training_circuits: list of circuits WITH measurements (for noisy runs)
        ideal_values: list of corresponding ideal expectation values (floats)
    """
    training_circuits = []
    ideal_values = []

    # Use the first subcircuit as template
    target_qc = base_subcircuits[0]

    print(f"Generating {num_samples} smart calibration variants...")

    for _ in range(num_samples):
        # 1. Copy the circuit and perturb parameters while keeping structure
        cal_qc = target_qc.copy()
        param_shifted = False

        # Iterate instructions safely and replace numeric params
        for i in range(len(cal_qc.data)):
            instruction = cal_qc.data[i]
            original_op = instruction.operation

            if getattr(original_op, 'params', None) and len(original_op.params) > 0:
                new_params = []
                for p in original_op.params:
                    if isinstance(p, (int, float, np.floating)):
                        shift = np.random.uniform(-0.2, 0.2)
                        new_params.append(float(p) + shift)
                        param_shifted = True
                    elif isinstance(p, (Parameter, ParameterExpression)):
                        sampled = float(np.random.uniform(0.1, 0.5))
                        new_params.append(sampled)
                        param_shifted = True
                    else:
                        try:
                            new_params.append(float(p))
                            param_shifted = True
                        except Exception:
                            new_params.append(p)

                # try to create a new op instance with new_params
                try:
                    new_op = original_op.__class__(*new_params)
                except Exception:
                    try:
                        new_op = original_op.copy()
                        new_op.params = new_params
                    except Exception:
                        new_op = original_op

                new_instruction = CircuitInstruction(new_op, instruction.qubits, instruction.clbits)
                cal_qc.data[i] = new_instruction

        # Fallback: if no parameters found, add a small rx
        if not param_shifted:
            cal_qc = cal_qc.copy()
            cal_qc.rx(np.random.uniform(0.1, 0.5), 0)

        # Build two versions:
        # - no-measure circuit for exact/Statevector evaluation
        # - measured circuit for noisy/shots-based execution
        cal_qc_no_meas = _strip_measurements(cal_qc)
        cal_qc_meas = _ensure_measured(cal_qc)

        # Compute ideal expectation on no-measure circuit (global Z^n)
        try:
            psi = Statevector.from_instruction(cal_qc_no_meas)
            from qiskit.quantum_info import Pauli
            op = Pauli('Z' * cal_qc_no_meas.num_qubits)
            ideal_val = float(psi.expectation_value(op).real)
        except Exception as e:
            # If something goes wrong, fallback to 0 and warn
            print(f"[Warning] Statevector ideal eval failed for a calibration circuit: {e}")
            ideal_val = 0.0

        training_circuits.append(cal_qc_meas)
        ideal_values.append(ideal_val)

    return training_circuits, ideal_values
