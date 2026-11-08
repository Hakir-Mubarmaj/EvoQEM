#!/usr/bin/env python3
"""
analyze_fidelity_crosscheck.py
- Loads dataset_counts.csv
- Compares each Ideal Count vs ALL Noisy Counts.
- Finds the best match and checks if it belongs to a different circuit.
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- Configuration ----------------
INPUT_CSV = 'test.csv'
OUTPUT_CSV = 'test_fidelity_analysis.csv'

# ---------------- Utility Functions ----------------

def calculate_hellinger_fidelity(counts1, counts2):
    """
    Calculates Hellinger Fidelity between two count distributions.
    Formula: F = (Sum(sqrt(p1 * p2)))^2
    """
    # 1. Total shots calculate kora
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())
    
    if shots1 == 0 or shots2 == 0:
        return 0.0

    # 2. Sob keys (basis states) eksathe kora
    all_keys = set(counts1.keys()) | set(counts2.keys())
    
    fidelity_sum = 0.0
    
    # 3. Probability calculate kore gun kora
    for key in all_keys:
        p1 = counts1.get(key, 0) / shots1
        p2 = counts2.get(key, 0) / shots2
        fidelity_sum += np.sqrt(p1 * p2)
        
    return fidelity_sum ** 2

# ---------------- Main Process ----------------

def main():
    print(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} file not found. Please run the generation script first.")
        return

    # Pre-parsing JSON to dictionaries to speed up the loop
    # Loop er bhitor bar bar json.loads korle slow hobe, tai agei list e nie nicchi
    ideal_dicts = [json.loads(x) for x in df['ideal_fidelity_counts']]
    noisy_dicts = [json.loads(x) for x in df['noisy_fidelity_counts']]
    circuit_ids = df['circuit_id'].tolist()
    
    results = []
    
    print(f"Processing {len(df)} circuits (comparing Ideal vs All Noisy)...")
    
    # Outer Loop: Protita Circuit er IDEAL count er jonno
    for i in tqdm(range(len(df)), desc="Analyzing"):
        target_ideal = ideal_dicts[i]
        target_id = circuit_ids[i]
        
        best_fidelity = -1.0
        best_match_id = -1
        
        # Inner Loop: Dataset er SOB Circuit er NOISY count er sathe tulona
        for j in range(len(df)):
            candidate_noisy = noisy_dicts[j]
            candidate_id = circuit_ids[j]
            
            # Fidelity calculate kora
            fid = calculate_hellinger_fidelity(target_ideal, candidate_noisy)
            
            # Maximum fidelity update kora
            if fid > best_fidelity:
                best_fidelity = fid
                best_match_id = candidate_id
        
        # Cross Match Logic
        # Jodi best_match_id ebong nijer id VINNO hoy -> True
        # Jodi best_match_id ebong nijer id SAME hoy -> False
        is_cross_match = (best_match_id != target_id)
        
        results.append({
            'circuit_id': target_id,
            'best_match_id': best_match_id,
            'highest_fidelity': round(best_fidelity, 6),
            'is_cross_match': is_cross_match
        })

    # DataFrame toiri kore Save kora
    result_df = pd.DataFrame(results)
    
    # Statistics print kora
    total_mismatches = result_df['is_cross_match'].sum()
    print(f"\nAnalysis Complete!")
    print(f"Total Circuits: {len(df)}")
    print(f"Cross Matches Found (Confusion): {total_mismatches}")
    print(f"Match Accuracy: {100 * (1 - total_mismatches/len(df)):.2f}%")
    
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()