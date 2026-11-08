#!/usr/bin/env python3
"""
cpdr_fixed_with_stats_expectation_style.py

CPDR pipeline with evaluation + figure identical in style to RobustPDRPipeline:
- mean, std, 95% CI of expectation values
- absolute error vs ideal
- paired t-test on errors
- fraction improved
- bootstrap CI
- boxplot of expectation values + ideal line + mean ± CI
"""

import os, csv, math, json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, NoiseModel

# -------------------------
# Output
# -------------------------
OUT_DIR = "csv"
os.makedirs(OUT_DIR, exist_ok=True)
PAIRED_CSV = os.path.join(OUT_DIR, "paired_results.csv")
FIG_FILE = os.path.join(OUT_DIR, "cpdr_stat_summary_expectation.png")

# -------------------------
# (All CPDR helper functions unchanged)
# -------------------------
_single_mul = {
    ('I','I'):(1,'I'), ('I','X'):(1,'X'), ('I','Y'):(1,'Y'), ('I','Z'):(1,'Z'),
    ('X','I'):(1,'X'), ('X','X'):(1,'I'), ('X','Y'):(1j,'Z'), ('X','Z'):(-1j,'Y'),
    ('Y','I'):(1,'Y'), ('Y','X'):(-1j,'Z'), ('Y','Y'):(1,'I'), ('Y','Z'):(1j,'X'),
    ('Z','I'):(1,'Z'), ('Z','X'):(1j,'Y'), ('Z','Y'):(-1j,'X'), ('Z','Z'):(1,'I'),
}
_anticommute_pairs = {('X','Y'),('Y','X'),('Y','Z'),('Z','Y'),('Z','X'),('X','Z')}

def pauli_mul(p1,p2):
    phase=1+0j; out=[]
    for a,b in zip(p1,p2):
        ph,r=_single_mul[(a,b)]
        phase*=ph; out.append(r)
    return phase,''.join(out)

def pauli_commutes(p1,p2):
    cnt=0
    for a,b in zip(p1,p2):
        if a!='I' and b!='I' and (a,b) in _anticommute_pairs:
            cnt+=1
    return (cnt%2)==0

def SPD_truncated_expectation(obs,gates,thetas,M):
    terms=[(obs,1+0j,0)]
    for P,theta in zip(reversed(gates),reversed(thetas)):
        new={}
        c,s=math.cos(theta),math.sin(theta)
        for sigma,coeff,sc in terms:
            if pauli_commutes(sigma,P):
                new[(sigma,sc)] = new.get((sigma,sc),0)+coeff
            else:
                new[(sigma,sc)] = new.get((sigma,sc),0)+coeff*c
                ph,ns=pauli_mul(P,sigma)
                new[(ns,sc+1)] = new.get((ns,sc+1),0)+coeff*(1j*s)*ph
        terms=[(s,new[(s,sc)],sc) for (s,sc) in new if sc<=M]
    exp=0
    for s,c,_ in terms:
        if all(x in ('I','Z') for x in s): exp+=c
    return float(np.real_if_close(exp))

def build_trotter_gate_lists(n,steps,th,tJ):
    g=[];t=[]
    for _ in range(steps):
        for i in range(n):
            P=['I']*n;P[i]='X';g.append(''.join(P));t.append(th)
        for i in range(n-1):
            P=['I']*n;P[i]=P[i+1]='Z';g.append(''.join(P));t.append(tJ)
    return g,t

def build_theta_grid():
    small=list(range(0,6)); large=list(range(54,60))
    idxs=small+large
    th=[i*math.pi/120 for i in idxs]
    tJ=[-j*math.pi/120 for j in idxs]
    return [(a,b) for a in th for b in tJ]

def build_trotter_qc(n,steps,th,tJ,meas=True):
    qc=QuantumCircuit(n)
    for _ in range(steps):
        for i in range(n): qc.rx(th,i)
        for i in range(n-1): qc.rzz(tJ,i,i+1)
    if meas: qc.measure_all()
    return qc

def make_depolarizing_noise_model(p1,p2):
    nm=NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1,1),['rx','u3'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2,2),['rzz','cx'])
    return nm

def counts_expectation_pauli(counts,pauli):
    shots=sum(counts.values()); exp=0
    for bitstr,c in counts.items():
        b=bitstr[::-1]; v=1
        for bit,p in zip(b,pauli):
            if p=='Z': v*=1 if bit=='0' else -1
        exp+=v*c
    return exp/shots

# -------------------------
# CPDR single run (unchanged)
# -------------------------
def cpdr_single_run(seed_base=0,n_qubits=2,steps=1,G_list=(1.0,1.2,1.6),
                    shots=8000,trunc_M=3,ridge_alpha=2e-5,
                    base_depol_1q=0.001,base_depol_2q=0.005):

    seed=seed_base
    def next_seed():
        nonlocal seed; s=seed; seed+=1; return s

    T=build_theta_grid()
    K=len(T)
    obs='Z'*n_qubits
    X=np.zeros((K,len(G_list))); y=np.zeros(K)

    sim=AerSimulator(method='density_matrix')

    for k,(th,tJ) in enumerate(T):
        g,t=build_trotter_gate_lists(n_qubits,steps,th,tJ)
        y[k]=SPD_truncated_expectation(obs,g,t,trunc_M)
        qc=build_trotter_qc(n_qubits,steps,th,tJ,True)
        tqc=transpile(qc,sim)
        for j,G in enumerate(G_list):
            nm=make_depolarizing_noise_model(base_depol_1q*G,base_depol_2q*G)
            res=sim.run(tqc,noise_model=nm,shots=shots,seed_simulator=next_seed()).result()
            X[k,j]=counts_expectation_pauli(res.get_counts(),obs)

    model=Ridge(alpha=ridge_alpha)
    model.fit(X,y)

    # target exact
    qc0=QuantumCircuit(n_qubits)
    qc0.rx(0.6,0);qc0.rx(0.8,1);qc0.cz(0,1)
    psi=Statevector.from_instruction(qc0)
    # compute exact <ZZ> manually (version-safe)
    exact = 0.0
    sv = psi.data
    for idx, amp in enumerate(sv):
        bits = format(idx, f'0{n_qubits}b')[::-1]
        v = 1
        for b in bits:
            v *= 1 if b == '0' else -1
        exact += v * (abs(amp) ** 2)
    exact = float(exact)

    qc=QuantumCircuit(n_qubits)
    qc.rx(0.6,0);qc.rx(0.8,1);qc.cz(0,1);qc.measure_all()
    tqc=transpile(qc,sim)

    nm0=make_depolarizing_noise_model(base_depol_1q,base_depol_2q)
    res0=sim.run(tqc,noise_model=nm0,shots=shots,seed_simulator=next_seed()).result()
    noisy=counts_expectation_pauli(res0.get_counts(),obs)

    feat=[]
    for G in G_list:
        nm=make_depolarizing_noise_model(base_depol_1q*G,base_depol_2q*G)
        r=sim.run(tqc,noise_model=nm,shots=shots,seed_simulator=next_seed()).result()
        feat.append(counts_expectation_pauli(r.get_counts(),obs))
    mit=float(model.predict(np.array(feat).reshape(1,-1))[0])

    return {'exact':exact,'noisy':noisy,'mitigated':mit}

# -------------------------
# Statistical validation (RobustPDR style)
# -------------------------
def run_statistical_validation_cpdr(n_repeats=30,seed_start=2000):

    if os.path.exists(PAIRED_CSV): os.remove(PAIRED_CSV)

    noisy_vals=[]; mit_vals=[]; exact=None

    for i in range(n_repeats):
        out=cpdr_single_run(seed_base=seed_start+i*1000)
        if exact is None: exact=out['exact']
        noisy_vals.append(out['noisy']); mit_vals.append(out['mitigated'])

        row={
            'run':i+1,'exact':out['exact'],
            'noisy':out['noisy'],'mitigated':out['mitigated'],
            'abs_err_noisy':abs(out['noisy']-exact),
            'abs_err_mitigated':abs(out['mitigated']-exact)
        }
        write_header=not os.path.exists(PAIRED_CSV)
        with open(PAIRED_CSV,'a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row.keys())
            if write_header: w.writeheader()
            w.writerow(row)

    noisy=np.array(noisy_vals); mit=np.array(mit_vals)
    ideal=exact

    err_raw=np.abs(noisy-ideal)
    err_mit=np.abs(mit-ideal)
    diffs=err_raw-err_mit

    def stats_ci(data):
        m=np.mean(data); s=np.std(data,ddof=1)
        sem=s/np.sqrt(len(data))
        h=sem*stats.t.ppf(0.975,len(data)-1)
        return m,s,h

    noisy_m,noisy_s,noisy_h=stats_ci(noisy)
    mit_m,mit_s,mit_h=stats_ci(mit)

    print("\n[Noisy]")
    print(f" Mean={noisy_m:.5f}, Std={noisy_s:.5f}, 95% CI=[{noisy_m-noisy_h:.5f},{noisy_m+noisy_h:.5f}]")
    print(f" Abs error={abs(noisy_m-ideal):.5f}")

    print("\n[CPDR]")
    print(f" Mean={mit_m:.5f}, Std={mit_s:.5f}, 95% CI=[{mit_m-mit_h:.5f},{mit_m+mit_h:.5f}]")
    print(f" Abs error={abs(mit_m-ideal):.5f}")

    t,p=stats.ttest_rel(err_mit,err_raw)
    frac=np.mean(diffs>0)

    rng=np.random.default_rng(123)
    boots=[rng.choice(diffs,len(diffs),True).mean() for _ in range(5000)]
    lo,hi=np.percentile(boots,[2.5,97.5])

    print("\nPaired t-test:",t,p)
    print("Frac improved:",frac)
    print("Bootstrap CI:",(lo,hi))

    # ---------- Figure (expectation values) ----------
    plt.figure(figsize=(10,6))
    plt.boxplot([noisy,mit],positions=[1,2],widths=0.4,patch_artist=True,
                boxprops=dict(facecolor='lightgray'))

    x1=np.random.normal(1,0.04,len(noisy))
    x2=np.random.normal(2,0.04,len(mit))
    plt.scatter(x1,noisy,alpha=0.6,color='red')
    plt.scatter(x2,mit,alpha=0.6,color='green')

    plt.axhline(ideal,color='blue',ls='--',lw=2,label=f'Ideal={ideal:.4f}')
    plt.errorbar(1,noisy_m,yerr=noisy_h,fmt='o',color='black',capsize=5)
    plt.errorbar(2,mit_m,yerr=mit_h,fmt='o',color='black',capsize=5)

    plt.xticks([1,2],['Noisy','CPDR'])
    plt.ylabel('<ZZ>')
    plt.title(f'CPDR statistical validation (N={len(noisy)})')
    plt.legend()
    plt.grid(axis='y',ls='--',alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_FILE,dpi=300)

    print("Saved figure:",FIG_FILE)

# -------------------------
if __name__=="__main__":
    run_statistical_validation_cpdr(n_repeats=100)
