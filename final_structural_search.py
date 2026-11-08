#!/usr/bin/env python3
"""
final_structural_search.py (Q1 Level Approach)
Uses Spectral Graph Theory to find structurally similar circuits from .qpy files.
Method:
1. Convert Quantum Circuit -> Directed Graph (DAG).
2. Compute Laplacian Spectrum (Eigenvalues) of the graph.
3. Use KNN on these spectral fingerprints to find the best match.
"""

import numpy as np
import pandas as pd
import networkx as nx
from qiskit import qpy, QuantumCircuit
from qiskit.converters import circuit_to_dag
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# কনফিগারেশন
DATASET_QPY = 'dataset_circuits.qpy'   # final_dg থেকে পাওয়া ১০০০ সার্কিট
DATASET_CSV = 'metadata_robust_final.csv' # জিনোম ডাটা পাওয়ার জন্য
TARGET_QPY = 'target_circuit.qpy'      # final_tc থেকে পাওয়া টার্গেট সার্কিট
TOP_K = 50  # কয়টা ক্লোজেস্ট সার্কিট চাই

def circuit_to_graph_fingerprint(qc: QuantumCircuit):
    """
    একটি কোয়ান্টাম সার্কিটকে গ্রাফে কনভার্ট করে তার Spectral Fingerprint বের করে।
    এটি Q1 জার্নালের জন্য একটি শক্তিশালী মেথডলজি।
    """
    # ১. সার্কিটকে DAG (Directed Acyclic Graph) এ কনভার্ট করা
    dag = circuit_to_dag(qc)
    
    # ২. NetworkX গ্রাফ তৈরি করা
    # আমরা কিউবিটদের ইন্টার‍্যাকশন গ্রাফ বানাবো
    G = nx.Graph()
    G.add_nodes_from(range(qc.num_qubits))
    
    # গেটগুলো চেক করে গ্রাফে এজ (Edge) যোগ করা
    for node in dag.topological_op_nodes():
        qargs = node.qargs
        if len(qargs) == 2: # ২-কিউবিট গেট (CNOT, CZ etc)
            u = qc.find_bit(qargs[0]).index
            v = qc.find_bit(qargs[1]).index
            
            # যদি অলরেডি এজ থাকে, তাহলে ওয়েট বাড়ানো (Stronger connection)
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
        
        # ১-কিউবিট গেটগুলোকে সেলফ-লুপ বা নোড ওয়েট হিসেবে ধরা যেতে পারে
        # সিমপ্লিফিকেশনের জন্য আমরা এখানে ইগনোর করছি অথবা নোড এট্রিবিউট বাড়াতে পারি
    
    # ৩. Spectral Fingerprint (Laplacian Eigenvalues) বের করা
    # এটি গ্রাফের টপোলজিকে সংখ্যার একটি ছোট লিস্টে কনভার্ট করে যা রোটেশন বা রি-অর্ডারিং এর প্রভাবিত হয় না।
    try:
        # Laplacian Matrix বের করা
        L = nx.laplacian_matrix(G, weight='weight').todense()
        # Eigenvalues বের করা
        eigenvalues = np.linalg.eigvalsh(L)
        # ছোট থেকে বড় সর্ট করা
        return np.sort(eigenvalues)
    except:
        # যদি গ্রাফ এম্পটি হয় (খুব ছোট সার্কিট)
        return np.zeros(qc.num_qubits)

def main():
    print("Loading datasets...")
    
    # ১. ডেটাসেট লোড করা
    with open(DATASET_QPY, 'rb') as fd:
        dataset_circuits = qpy.load(fd)
    
    df_meta = pd.read_csv(DATASET_CSV)
    
    # ২. টার্গেট সার্কিট লোড করা
    with open(TARGET_QPY, 'rb') as fd:
        # qpy.load লিস্ট রিটার্ন করে, তাই [0] নিচ্ছি
        target_circuit = qpy.load(fd)[0] 
    
    print(f"Processing {len(dataset_circuits)} circuits for Spectral Analysis...")
    
    # ৩. ফ ফিচার এক্সট্রাকশন (Spectral Fingerprinting)
    dataset_fingerprints = []
    for qc in dataset_circuits:
        fp = circuit_to_graph_fingerprint(qc)
        dataset_fingerprints.append(fp)
    
    target_fingerprint = circuit_to_graph_fingerprint(target_circuit)
    
    # ডেটা শেপ ঠিক করা (Sklearn এর জন্য)
    X = np.array(dataset_fingerprints)
    y = np.array([target_fingerprint])
    
    # ৪. KNN মডেল চালানো (Metric: Manhattan or Euclidean distance on Spectrum)
    # Spectral ভ্যালুর জন্য Manhattan (L1) ভালো কাজ করে
    knn = NearestNeighbors(n_neighbors=TOP_K, metric='manhattan')
    knn.fit(X)
    
    distances, indices = knn.kneighbors(y)
    
    print(f"\n--- Top {TOP_K} Structurally Similar Circuits Found ---")
    
    suggestions = []
    
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        matched_row = df_meta.iloc[idx]
        
        # জিনোম এবং ইমপ্রুভমেন্ট ডাটা নিয়ে আসা
        suggestion = {
            'rank': rank + 1,
            'similarity_score': 1 / (1 + dist), # ডিসটেন্স কম মানে সিমিলারিটি বেশি
            'matched_id': idx,
            'suggested_genome': matched_row['best_genome'],
            'original_improvement': matched_row['improvement']
        }
        suggestions.append(suggestion)
        
        print(f"Rank #{rank+1} | Spectral Dist: {dist:.4f}")
        print(f"  -> Match ID: {idx}")
        print(f"  -> Genome: {matched_row['best_genome']}")
        print(f"  -> Proven Improvement: {matched_row['improvement']:.4f}")
        print("-" * 40)

    # ৫. রেজাল্ট সেভ করা (GA তে ব্যবহারের জন্য)
    pd.DataFrame(suggestions).to_csv('structural_suggestions.csv', index=False)
    print("\nBest suggestions saved to 'structural_suggestions.csv' for Genetic Algorithm.")

if __name__ == '__main__':
    main()