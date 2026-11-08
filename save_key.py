from qiskit_ibm_runtime import QiskitRuntimeService

# Replace with your actual key from the IBM Quantum Dashboard
MY_TOKEN = "Dj5oYygfQHEgRAwrfHdm4v1Pj0Xez_Tc9bNcWr17wPgx" 

# FIXED: Changed channel to 'ibm_quantum_platform' for newer Qiskit versions
try:
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform", 
        token=MY_TOKEN, 
        overwrite=True
    )
    print("SUCCESS: Account saved! You can now run the verification script.")
except Exception as e:
    print(f"FAILED: {e}")