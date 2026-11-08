"""
q1_hardware_verify_level0.py
----------------------------
SCIENTIFIC VERIFICATION (RAW BASELINE MODE).

Changes:
1. Transpiles the Raw Target to Hardware at Level 0 (Naive/Raw).
   - This creates a noisier, longer baseline circuit.
2. Applies EvoQEM genes to this SAME raw circuit.

Hypothesis: The 'Optimized' version should survive the increased depth
better than the 'Baseline' due to the genes.
"""

import pandas as pd
import json
import sys
import numpy as np

from qiskit import qpy, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

# --- IMPORT USER DEFINITIONS ---
try:
    from final_dg import GENE_MAP
    print("[INFO] Successfully imported GENE_MAP")
except ImportError:
    print("[CRITICAL] 'final_dg.py' not found.")
    sys.exit(1)

# --- CONFIGURATION ---
TARGET_QPY = 'target_circuit.qpy'
BEST_RESULT_CSV = 'ga_best_result.csv'
BACKEND_NAME = 'ibm_fez' 

def apply_genome_logic(qc, genome):
    """Applies genome to a circuit."""
    temp_qc = qc.copy()
    for gene_name in genome:
        if gene_name in GENE_MAP:
            try:
                temp_qc = GENE_MAP[gene_name](temp_qc)
            except Exception as e:
                print(f"[WARN] Gene '{gene_name}' skipped: {e}")
    return temp_qc

def run_submission():
    print("\n" + "="*60)
    print("   SCIENTIFIC VERIFICATION (LEVEL 0 BASELINE)   ")
    print("="*60 + "\n")

    # 1. Connect
    service = QiskitRuntimeService()
    try:
        backend = service.backend(BACKEND_NAME)
        print(f"[SELECTED] Backend: {backend.name}")
    except:
        print(f"[CRITICAL] Backend {BACKEND_NAME} not found.")
        sys.exit(1)

    # 2. Load Data
    try:
        with open(TARGET_QPY, 'rb') as f:
            qc_raw = qpy.load(f)[0]
        
        df = pd.read_csv(BEST_RESULT_CSV)
        best_genome = json.loads(df.iloc[0]['best_genome'])
        print(f"Learned Genome: {best_genome}")
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return

    # ---------------------------------------------------------
    # 3. PRE-PROCESSING (Modified to Level 0)
    # ---------------------------------------------------------
    print("\n--- Step 1: Generating Canonical Physical Circuit ---")
    print("Transpiling raw circuit with Optimization Level 0 (Naive/Raw)...")
    
    if qc_raw.num_clbits == 0:
        qc_raw.measure_all()

    # CHANGED HERE: Level 0 preserves the raw structure (and inefficiencies)
    # This gives the genes more 'mess' to clean up.
    qc_physical_canonical = transpile(qc_raw, backend=backend, optimization_level=0)
    
    print(f"Canonical Depth: {qc_physical_canonical.depth()}")
    print(f"Canonical Gates: {sum(qc_physical_canonical.count_ops().values())}")

    # ---------------------------------------------------------
    # 4. BUILD VARIANTS
    # ---------------------------------------------------------
    print("\n--- Step 2: Branching Variants ---")

    # A. Baseline (The Naive Circuit)
    qc_baseline = transpile(qc_physical_canonical, backend=backend, optimization_level=0)
    qc_baseline.name = "Baseline_L0"

    # B. EvoQEM Optimized (Naive Circuit + Genes)
    print(f"Applying genes to Canonical Circuit...")
    qc_opt_struct = apply_genome_logic(qc_physical_canonical, best_genome)
    
    # Final bind
    qc_optimized = transpile(qc_opt_struct, backend=backend, optimization_level=0)
    qc_optimized.name = "EvoQEM_Optimized_L0"

    # ---------------------------------------------------------
    # 5. FINAL CHECKS
    # ---------------------------------------------------------
    print(f"\n--- Final Comparison Stats ---")
    print(f"Baseline Depth:  {qc_baseline.depth()}")
    print(f"Optimized Depth: {qc_optimized.depth()}")
    
    if qc_baseline.layout.final_layout != qc_optimized.layout.final_layout:
        print("[WARN] Layouts differ! Comparison might be slightly noisy.")
    else:
        print("[INFO] Physical layouts match. Comparison is scientifically valid.")

    # ---------------------------------------------------------
    # 6. SUBMISSION
    # ---------------------------------------------------------
    print("\n[SUBMITTING]...")
    try:
        sampler = SamplerV2(mode=backend)
        job = sampler.run([qc_baseline, qc_optimized])
        print(f"Job ID: {job.job_id()}")
        print("Check status: https://quantum.ibm.com/jobs")
    except Exception as e:
        print(f"Submission Error: {e}")

if __name__ == "__main__":
    run_submission()