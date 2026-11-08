#!/usr/bin/env python3
"""
final_ga_random.py
Control Script: Optimizes the genome for 'target_circuit.qpy' using a Genetic Algorithm.
RANDOM MODE: Starts with completely random genomes (No Transfer Learning).
FIXED: Corrected safe_transpile and compute_fidelity arguments.
"""

import pandas as pd
import numpy as np
import json
import ast
import random
import warnings
from tqdm import tqdm

# Qiskit Imports
from qiskit import qpy
from qiskit_aer import AerSimulator

# --- IMPORTING FROM YOUR DATA GENERATION SCRIPT ---
try:
    from final_dg import (
        GENE_MAP, 
        get_noise_model, 
        compute_fidelity, 
        safe_transpile,
        N_QUBITS,
        GENOME_LENGTH
    )
except ImportError:
    raise ImportError("Error: 'final_dg.py' not found! Make sure it is in the same folder.")

# ---------------- CONFIGURATION ----------------
TARGET_QPY_FILE = 'target_circuit.qpy'  
OUTPUT_CSV = 'ga_random_result.csv'     

# GA Parameters
POPULATION_SIZE = 20    
GENERATIONS = 200        
MUTATION_RATE = 0.2     
ELITISM_COUNT = 1       

# Simulator Setup
SIMULATOR = AerSimulator(method='density_matrix')
NOISE_MODEL = get_noise_model()  

def get_target_circuit():
    """Loads the specific target circuit we want to optimize."""
    with open(TARGET_QPY_FILE, 'rb') as fd:
        qc = qpy.load(fd)[0]
    return qc

def calculate_fitness(genome, target_qc):
    """
    Applies the genome (DD Sequence) to the circuit and measures fidelity.
    """
    # 1. Apply Genome to create a candidate circuit
    # We copy the target first, then modify it
    qc_candidate = target_qc.copy()
    
    for gene_name in genome:
        if gene_name in GENE_MAP:
            qc_candidate = GENE_MAP[gene_name](qc_candidate)
    
    # 2. Transpile
    # CRITICAL FIX: safe_transpile only takes (qc, optimization_level)
    # We use opt_level=0 to ensure our DD sequences (like XY4) aren't optimized away.
    try:
        noisy_qc = safe_transpile(qc_candidate, optimization_level=0)
    except Exception as e:
        return 0.0 # Penalty for invalid circuits
    
    if noisy_qc is None:
        return 0.0
    
    # 3. Compute Fidelity
    # CRITICAL FIX: compute_fidelity takes (qc, noise_model, simulator)
    fid = compute_fidelity(noisy_qc, NOISE_MODEL, SIMULATOR)
    return fid

def crossover(parent1, parent2):
    """Single Point Crossover"""
    if len(parent1) < 2: return parent1 # Safety
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(genome):
    """Randomly flips one gene in the sequence"""
    idx = random.randint(0, len(genome) - 1)
    new_gene = random.choice(list(GENE_MAP.keys()))
    genome[idx] = new_gene
    return genome

def main():
    print(f"--- Starting GA (RANDOM INITIALIZATION) ---")
    
    # Load Target
    try:
        target_qc = get_target_circuit()
    except FileNotFoundError:
        print(f"Error: {TARGET_QPY_FILE} not found. Run final_tc.py first.")
        return

    # Calculate Baseline (No Mitigation)
    baseline_seq = ["DoNothing"] * GENOME_LENGTH
    baseline_fid = calculate_fitness(baseline_seq, target_qc)
    print(f"Baseline (DoNothing) Fidelity: {baseline_fid:.4f}")

    # ==========================================================
    # 1. Initialize Population (RANDOM MODE)
    # ==========================================================
    print(f"Initializing population with {POPULATION_SIZE} random genomes...")
    
    population = []
    possible_genes = list(GENE_MAP.keys())
    
    for _ in range(POPULATION_SIZE):
        # Create a random sequence of genes
        random_genome = [random.choice(possible_genes) for _ in range(GENOME_LENGTH)]
        population.append(random_genome)
        
    # ==========================================================
    # 2. Evolution Loop
    # ==========================================================
    best_global_genome = None
    best_global_fid = -1.0
    
    for gen in range(GENERATIONS):
        scores = []
        
        # Evaluate current population
        for genome in population:
            fid = calculate_fitness(genome, target_qc)
            scores.append((genome, fid))
            
            if fid > best_global_fid:
                best_global_fid = fid
                best_global_genome = genome
        
        # Sort by fidelity (Descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        avg_fid = np.mean([s[1] for s in scores])
        print(f"Gen {gen+1}/{GENERATIONS} | Max: {scores[0][1]:.4f} | Avg: {avg_fid:.4f}")
        
        # Selection & Elitism
        # Keep the absolute best
        new_population = [x[0] for x in scores[:ELITISM_COUNT]]
        
        # Parents pool (Top 50%)
        parents_pool = [x[0] for x in scores[:len(scores)//2]] 
        
        # Breeding
        while len(new_population) < POPULATION_SIZE:
            p1 = random.choice(parents_pool)
            p2 = random.choice(parents_pool)
            
            child = crossover(p1, p2)
            
            if random.random() < MUTATION_RATE:
                child = mutate(child)
                
            new_population.append(child)
            
        population = new_population

    # ==========================================================
    # 3. Save Results
    # ==========================================================
    print(f"\n--- Random Init Optimization Complete ---")
    print(f"Baseline Fidelity:     {baseline_fid:.4f}")
    print(f"Optimized Fidelity:    {best_global_fid:.4f}")
    print(f"Net Improvement:       {best_global_fid - baseline_fid:.4f}")
    print(f"Best Genome Sequence:  {best_global_genome}")

    result_data = {
        'circuit_file': [TARGET_QPY_FILE],
        'baseline_fidelity': [baseline_fid],
        'optimized_fidelity': [best_global_fid],
        'improvement': [best_global_fid - baseline_fid],
        'best_genome': [json.dumps(best_global_genome)]
    }
    
    df_res = pd.DataFrame(result_data)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()