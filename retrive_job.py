import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

# --- CONFIGURATION ---
JOB_ID = 'd4tevj5eastc73cg2gs0'
TRUE_TARGET_STATE = '110'  # From your Ideal Simulation

print(f"Fetching results for Job ID: {JOB_ID}...")
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    instance='crn:v1:bluemix:public:quantum-computing:us-east:a/894079ffd0f0488cb7318d6dfde07e56:aaa6f51e-7de4-4099-87fa-3f0d58fcff00::'
)
job = service.job(JOB_ID)
result = job.result()

def get_prob_distribution(pub_result):
    counts = pub_result.data.meas.get_counts()
    total = sum(counts.values())
    return {state: count / total for state, count in counts.items()}

baseline_probs = get_prob_distribution(result[0])
xy4_probs = get_prob_distribution(result[1]) # This was Baseline proxy
optimized_probs = get_prob_distribution(result[2])

print("\n--- PROBABILITY OF TRUE TARGET |110> ---")
p_base = baseline_probs.get(TRUE_TARGET_STATE, 0)
p_opt = optimized_probs.get(TRUE_TARGET_STATE, 0)

print(f"Baseline:  {p_base:.4f}")
print(f"Optimized: {p_opt:.4f}")

# Plotting
labels = ['Baseline', 'EvoQEM Optimized']
values = [p_base, p_opt]
colors = ['gray', 'green']

plt.figure(figsize=(6, 5))
bars = plt.bar(labels, values, color=colors, edgecolor='black', width=0.5)
plt.ylabel(f"Probability of Correct State |{TRUE_TARGET_STATE}>")
plt.title(f"Error Mitigation Performance\n(True Target: {TRUE_TARGET_STATE})")
plt.ylim(0, max(values)*1.3)

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f"{bar.get_height():.4f}", ha='center', fontsize=12, fontweight='bold')

plt.savefig("q1_real_truth_plot.png", dpi=300)
print("\nPlot saved as 'q1_real_truth_plot.png'.")

if p_opt > p_base:
    print("\n[CONCLUSION] SUCCESS: Optimized circuit recovered the True State better!")
else:
    print("\n[CONCLUSION] WARNING: Optimization did not improve the True State probability.")