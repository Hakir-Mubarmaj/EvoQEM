import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter, ParameterExpression, CircuitInstruction

# Reproducibility seed (optional) -- you can change or remove if you want true randomness
np.random.seed(42)


def generate_calibration_circuits(base_subcircuits, num_samples=20):
    """
    Smart Calibration: Modifies EXISTING angles instead of adding new gates.
    This preserves the circuit structure and prevents SWAP explosion during transpilation.

    Args:
        base_subcircuits (list): List containing the base circuit to perturb.
        num_samples (int): Number of training samples to generate.

    Returns:
        tuple: (training_circuits, ideal_values)
    """
    training_circuits = []
    ideal_values = []

    # We take the first subcircuit as the template
    target_qc = base_subcircuits[0]

    print(f"Generating {num_samples} smart calibration variants...")

    for _ in range(num_samples):
        # 1. Copy the circuit
        cal_qc = target_qc.copy()

        # 2. Perturb parameters while keeping structure
        param_shifted = False

        # Iterate by index to replace instructions safely
        for i in range(len(cal_qc.data)):
            instruction = cal_qc.data[i]
            original_op = instruction.operation

            # Check if the gate has parameters (angles)
            if getattr(original_op, 'params', None) and len(original_op.params) > 0:
                new_params = []

                for p in original_op.params:
                    # Numeric floats/ints -> shift
                    if isinstance(p, (int, float, np.floating)):
                        shift = np.random.uniform(-0.2, 0.2)
                        new_params.append(float(p) + shift)
                        param_shifted = True

                    # Symbolic Parameter or ParameterExpression -> sample a concrete value
                    elif isinstance(p, (Parameter, ParameterExpression)):
                        sampled = float(np.random.uniform(0.1, 0.5))
                        new_params.append(sampled)
                        param_shifted = True

                    else:
                        # Attempt to coerce to float (handles some ParameterExpression-like objects)
                        try:
                            new_params.append(float(p))
                            param_shifted = True
                        except Exception:
                            # Last resort: keep as-is (rare)
                            new_params.append(p)

                # Try to build a new operation instance of the same class
                try:
                    # Many Gate classes accept parameters in constructor; this works for RX, RY, RZ etc.
                    new_op = original_op.__class__(*new_params)
                except Exception:
                    # Fallback: try copy-and-assign params
                    try:
                        new_op = original_op.copy()
                        new_op.params = new_params
                    except Exception:
                        # If we cannot safely construct a new op, skip modifying this instruction
                        new_op = original_op

                # Replace the instruction in the new circuit
                new_instruction = CircuitInstruction(new_op, instruction.qubits, instruction.clbits)
                cal_qc.data[i] = new_instruction

        # Fallback: if no parameters found (e.g., only H, CNOT), add a dummy rotation
        if not param_shifted:
            cal_qc = cal_qc.copy()
            cal_qc.rx(np.random.uniform(0.1, 0.5), 0)

        training_circuits.append(cal_qc)

        # 3. Calculate Ideal Expectation Value (<ZZ...Z>) for this perturbed circuit
        psi = Statevector.from_instruction(cal_qc)

        from qiskit.quantum_info import Pauli
        op = Pauli('Z' * cal_qc.num_qubits)
        ideal_val = psi.expectation_value(op).real
        ideal_values.append(ideal_val)

    return training_circuits, ideal_values
