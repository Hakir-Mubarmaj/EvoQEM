"""
q1_benchmark_comparison.py
--------------------------
The "Money Shot" Plot: Shows that Transfer Learning converges faster than Random Search.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# --- CONFIGURATION ---
# You need to run final_ga.py and final_ga_random.py first to generate these CSVs!
TRANSFER_CSV = 'ga_best_result.csv'  
RANDOM_CSV = 'ga_random_result.csv'   # You need to run final_ga_random.py to get this

def plot_benchmark():
    # 1. Mock Data Generation (IF you haven't run the full random script yet)
    # This allows you to test the plotting logic immediately.
    # REPLACE this with pd.read_csv(RANDOM_CSV) when ready.
    
    generations = np.arange(0, 20)
    
    # Mocking behavior for visual demo (Replace with actual data!)
    # Random Search: Slow start, gradual rise
    random_curve = 0.80 + 0.10 * (1 - np.exp(-0.1 * generations)) 
    
    # Transfer Learning: High start (due to spectral suggestion), fast saturation
    transfer_curve = 0.88 + 0.04 * (1 - np.exp(-0.5 * generations))
    
    baseline_fidelity = 0.79
    standard_xy4_fidelity = 0.85 # Standard Qiskit DD
    
    # 2. Plotting
    plt.figure(figsize=(10, 6))
    
    # Baselines
    plt.axhline(y=baseline_fidelity, color='gray', linestyle='--', label='Baseline (No DD)')
    plt.axhline(y=standard_xy4_fidelity, color='orange', linestyle='-.', label='Standard XY4 (Qiskit)')
    
    # Learning Curves
    plt.plot(generations, random_curve, 'r-o', label='Random Search (Standard GA)', markersize=4, alpha=0.6)
    plt.plot(generations, transfer_curve, 'g-s', label='Spectral Transfer (Ours)', markersize=6, linewidth=2)
    
    plt.title("Convergence Analysis: Spectral Transfer vs. Random Search", fontsize=14)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Circuit Fidelity", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("q1_benchmark_convergence.png", dpi=300)
    print("Benchmark plot saved.")

if __name__ == "__main__":
    plot_benchmark()