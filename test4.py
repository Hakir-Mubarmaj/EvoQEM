# paired_compare.py
import numpy as np
import pandas as pd
from scipy import stats
from test3 import cpdr_zne_pipeline   # adjust import paths
from test2 import run_cpdr_with_qpdr  # adjust import names
# both functions must accept seed parameter (see instructions)

def single_pair_run(seed, cpdr_kwargs=None, qpdr_kwargs=None):
    if cpdr_kwargs is None: cpdr_kwargs = {}
    if qpdr_kwargs is None: qpdr_kwargs = {}
    # run CPDR pipeline with specified seed
    out_cpdr = cpdr_zne_pipeline(seed=seed, **cpdr_kwargs)
    cpdr_est = float(out_cpdr['target']['mitigated'])
    exact = float(out_cpdr['target']['exact'])

    # run QPDR pipeline but ensure it uses same seed for simulator runs
    out_qpdr = run_cpdr_with_qpdr(seed=seed, **qpdr_kwargs)
    qpdr_est = float(out_qpdr['qpdr_mitigated'])
    # Note: ensure run_cpdr_with_qpdr returns 'exact' same as cpdr pipeline or recompute

    return {'seed':seed, 'exact':exact, 'cpdr':cpdr_est, 'qpdr':qpdr_est,
            'err_cpdr':abs(cpdr_est-exact), 'err_qpdr':abs(qpdr_est-exact),
            'noisy': out_cpdr['target']['noisy']}

def paired_experiment(N=30, cpdr_kwargs=None, qpdr_kwargs=None):
    results = []
    for seed in range(1, N+1):
        print(f"Running pair {seed}/{N} ...", end='\r')
        res = single_pair_run(seed, cpdr_kwargs, qpdr_kwargs)
        results.append(res)
    df = pd.DataFrame(results)
    df.to_csv('paired_results.csv', index=False)
    # summary stats
    err_cpdr = df['err_cpdr'].values
    err_qpdr = df['err_qpdr'].values
    noisy_err = np.abs(df['noisy'].values - df['exact'].values)
    print("\nSummary:")
    def summarize(name, arr):
        mean = arr.mean(); std=arr.std(ddof=1)
        sem = std/np.sqrt(len(arr)); h = sem * stats.t.ppf(0.975, len(arr)-1)
        print(f"{name}: mean={mean:.6f} ± {h:.6f} (n={len(arr)})")
    summarize('Noisy err', noisy_err)
    summarize('CPDR err', err_cpdr)
    summarize('QPDR err', err_qpdr)
    # paired t-test (noisy vs qpdr), (noisy vs cpdr), (cpdr vs qpdr)
    print("paired t (noisy vs qm):", stats.ttest_rel(noisy_err, err_qpdr))
    print("paired t (noisy vs cpdr):", stats.ttest_rel(noisy_err, err_cpdr))
    print("paired t (cpdr vs qpdr):", stats.ttest_rel(err_cpdr, err_qpdr))
    # fraction improved
    print("Frac improved (QPDR vs noisy):", np.mean(noisy_err > err_qpdr))
    print("Frac improved (CPDR vs noisy):", np.mean(noisy_err > err_cpdr))
    return df

if __name__ == "__main__":
    df = paired_experiment(N=30)
    print("Saved paired_results.csv")
