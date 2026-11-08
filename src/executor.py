# executor.py
# Robust VirtualExecutor with fallbacks for different qiskit runtime/backends

from qiskit_ibm_runtime.fake_provider import FakeFez, FakeSherbrooke
from qiskit import transpile
from qiskit.circuit import Measure
import numpy as np

# Qiskit Runtime Imports for Real Hardware (SamplerV2)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli


class VirtualExecutor:
    def __init__(self, backend_name='fez'):
        # Select a fake backend for testing. On real hardware you should replace this by the real service backend.
        if backend_name == 'fez':
            self.backend = FakeFez()
        else:
            # default fallback
            self.backend = FakeSherbrooke()

        print(f"Executor initialized with: {getattr(self.backend, 'name', str(self.backend))}")

    def _add_measurements_if_needed(self, qc):
        # Return a new circuit that has measurements on all qubits.
        # Use non-inplace measure_all if available; otherwise construct manually.
        try:
            # Many qiskit versions support measure_all(inplace=False)
            qc_meas = qc.measure_all(inplace=False)
            return qc_meas
        except TypeError:
            # Older qiskit versions may not accept inplace flag; use copy + measure_all
            qc_copy = qc.copy()
            qc_copy.measure_all()
            return qc_copy
        except Exception:
            # Last resort: build a new circuit and append measurements manually
            qc_copy = qc.copy()
            try:
                n = qc_copy.num_qubits
                for q in range(n):
                    qc_copy.measure(q, q)
            except Exception:
                pass
            return qc_copy

    def _transpile(self, circuits):
        try:
            transpiled = transpile(circuits, self.backend, optimization_level=3, layout_method='sabre')
            return transpiled
        except Exception:
            # Fallback to basic transpile without layout method
            try:
                transpiled = transpile(circuits, self.backend, optimization_level=1)
                return transpiled
            except Exception:
                # If transpile fails, return original circuits
                return circuits

    def run_mitigation_experiments(self, circuits, shots=4000):
        """
        Runs circuits on the noisy backend and calculates raw expectation values.
        Automatically adds measurements to the circuits if missing.

        Args:
            circuits (list): List of QuantumCircuits (without measurements usually).
            shots (int): Number of shots.

        Returns:
            list: List of noisy expectation values <ZZ..Z>.
        """

        circuits_with_meas = [self._add_measurements_if_needed(qc) for qc in circuits]

        print(f"Transpiling {len(circuits_with_meas)} circuits with optimization...")
        transpiled_circuits = self._transpile(circuits_with_meas)

        # 3. Run job
        try:
            job = self.backend.run(transpiled_circuits, shots=shots)
            result = job.result()
        except AttributeError as e:
            # backend.run not available on some objects / versions
            raise RuntimeError("Backend does not expose 'run' API in this environment.\n"
                               "Please run against a proper IBM runtime Service backend or adapt this wrapper.")
        except Exception as e:
            raise RuntimeError(f"Failed to run circuits on backend: {e}")

        expvals = []
        for i in range(len(circuits)):
            try:
                counts = result.get_counts(i)
            except Exception as e:
                print(f"Error getting counts for circuit {i}: {e}")
                expvals.append(0.0)
                continue

            # Calculate Expectation Value <ZZ...Z> from counts
            val = 0
            total_counts = 0

            for state, count in counts.items():
                cleaned_state = state.replace(" ", "")
                ones = cleaned_state.count('1')

                if ones % 2 == 0:
                    val += count
                else:
                    val -= count
                total_counts += count

            if total_counts > 0:
                expvals.append(val / total_counts)
            else:
                expvals.append(0.0)

        return expvals


    
class RealHardwareExecutor:
    """
    Executor specifically designed for IBM Quantum Hardware (e.g., ibm_fez).
    Uses Qiskit Runtime Primitives (SamplerV2).
    """
    def __init__(self, backend_name='ibm_fez'):
        print(f"   [Init] Connecting to IBM Quantum Service...")
        try:
            self.service = QiskitRuntimeService()
            self.backend = self.service.backend(backend_name)
            print(f"   [Success] Connected to Real Backend: {self.backend.name}")
        except Exception as e:
            print(f"   [Error] Could not connect to backend: {e}")
            self.backend = None

    def run_mitigation_experiments(self, circuits, shots=4000):
        if self.backend is None:
            print("   [Error] No backend connection. Returning Zeros.")
            return [0.0] * len(circuits)

        # 1. Add Measurements explicitly (CRITICAL FIX)
        # Without this, the hardware runs but returns no data, causing IndexError.
        circuits_with_meas = []
        for qc in circuits:
            # Check if measurements already exist to avoid double measuring
            if qc.num_clbits == 0:
                qc_meas = qc.measure_all(inplace=False)
            else:
                qc_meas = qc
            circuits_with_meas.append(qc_meas)

        # 2. Transpile using PassManager
        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        isa_circuits = pm.run(circuits_with_meas)

        # Ensure isa_circuits is iterable/list even if one circuit
        if not isinstance(isa_circuits, list):
            isa_circuits = [isa_circuits]

        # 3. Run using Sampler V2
        sampler = Sampler(mode=self.backend)
        
        # --- FIX: Create proper PUBs (Primitive Unified Blocs) ---
        # Format: (circuit, parameter_values, shots)
        # We assume circuits are fully bound (no parameters needed), so None for params.
        pubs = [(c, None, shots) for c in isa_circuits]
        
        try:
            # Send the batch of PUBs
            job = sampler.run(pubs)
            result = job.result()
            
            # 4. Process Results
            expvals = []
            
            # Iterate over each PUB result
            for pub_result in result:
                # Retrieve measurement data safely
                try:
                    data = pub_result.data.meas
                except AttributeError:
                    # Fallback for dynamic register names if 'meas' is missing
                    # This retrieves the first available register in the DataBin
                    try:
                        # DataBin structure allows attribute access or via __dict__ in some versions
                        first_attr = list(pub_result.data.keys())[0]
                        data = getattr(pub_result.data, first_attr)
                    except (IndexError, AttributeError):
                         print("   [Error] No measurement data found in result.")
                         expvals.append(0.0)
                         continue
                
                counts = data.get_counts()
                
                val = 0
                total_counts = 0
                for state, count in counts.items():
                    # Calculate parity (number of 1s)
                    # Note: state is a bitstring
                    ones = state.count('1')
                    if ones % 2 == 0:
                        val += count
                    else:
                        val -= count
                    total_counts += count
                
                if total_counts > 0:
                    expvals.append(val / total_counts)
                else:
                    expvals.append(0.0)
            
            return expvals

        except Exception as e:
            print(f"   [Error] Job execution failed: {e}")
            import traceback
            traceback.print_exc()
            return [0.0] * len(circuits)
