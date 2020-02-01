import matplotlib.pyplot as plt
from tcorex.experiments import data as data_tools, baselines_cov_est
from tcorex.experiments.misc import make_sure_path_exists
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import os
import pickle
import json
from tcorex import TCorex


m = 32
p = 128

true_clusters = np.arange(m).repeat(p // m, axis=0)

methods = [
    (baselines_cov_est.Diagonal(name='Diagonal'), {}),
    (baselines_cov_est.LedoitWolf(name='Ledoit-Wolf'), {}),
    (baselines_cov_est.PCA(name='PCA'), {}),
    (baselines_cov_est.SparsePCA(name='SparsePCA'), {}),
    (baselines_cov_est.FactorAnalysis(name='Factor Analysis'), {}),
    (baselines_cov_est.GraphLasso(name='Graphical LASSO (sklearn)'), {}),
    (baselines_cov_est.LinearCorex(name='Linear CorEx'), {}),
    (baselines_cov_est.TCorex(TCorex, name='T-CorEx'), {'nv': p}),
    (baselines_cov_est.LVGLASSO(name='LVGLASSO'), {})
]

n_runs = 5

file_path = f'scores_n_m{m}.pkl'
if os.path.isfile(file_path):
    with open(file_path, 'rb') as f:
        scores_per_algo = pickle.load(f)
else:
    scores_per_algo = {}

n_min = 3
n_max = 8
r = list(range(n_min, n_max + 1))
n_experiments = len(r)
for k, i in enumerate(r):
    n = 2 ** i
    print("")
    print(f"Experiment {k+1} of {n_experiments}: n=2^{i}={n}")

    with open(f'outputs/blessing/best/blessing_experiment_n{n}_p{p}_m{m}.results.json', 'r') as f:
        results = json.load(f)

    for j in range(0, n_runs):

        skip = True
        for (method, _) in methods:
            name = method.name
            if name not in scores_per_algo \
                    or str(i) not in scores_per_algo[name] \
                    or len(scores_per_algo[name][str(i)]) <= j:
                skip = False
                break

        if skip:
            print(f"Skipping run {j+1} of {n_runs} for n=2^{i}={n}")
            continue

        print("")
        print(f"Starting run {j+1} of {n_runs} for n=2^{i}={n}")

        data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=n + 1000, snr=0.1,
                                                            num_extra_parents=0,
                                                            num_correlated_zs=0,
                                                            random_scale=False)

        train_data, test_data = train_test_split(data, test_size=1000, shuffle=True)

        for (method, params) in methods:
            best_score = 0
            name = method.name
            try:
                if name in scores_per_algo \
                        and str(i) in scores_per_algo[name] \
                        and len(scores_per_algo[name][str(i)]) > j:
                    print(f"Skipping {name}")
                    continue
                else:
                    print(f"Doing {name}")
                best_score, _, _, _, _, best_ari = method.select([train_data], [test_data], results[name]['best_params'] if name in results else params, verbose=False)
            except Exception:
                pass

            if name not in scores_per_algo:
                scores_per_algo[name] = {}
            if str(i) not in scores_per_algo[name]:
                scores_per_algo[name][str(i)] = []
            scores_per_algo[name][str(i)].append(best_score)

        with open(file_path, 'wb') as f:
            pickle.dump(scores_per_algo, f)

with open(file_path, 'wb') as f:
    pickle.dump(scores_per_algo, f)


scores_per_algo = {algo: {k: np.array([x[1] for x in v]) for k, v in itertools.groupby(results.items(), key=lambda entry: int(entry[0]))}
                   for algo, results in scores_per_algo.items()}

score_means = {algo: list(map(lambda x: [x[0], x[1].mean()], agg.items())) for algo, agg in scores_per_algo.items()}
score_stds = {algo: list(map(lambda x: [x[0], x[1].std()], agg.items())) for algo, agg in scores_per_algo.items()}

print(score_means)
print(score_stds)

plt.figure()
for algo, results in score_means.items():
    scores = np.array(results)
    stds = np.array(score_stds[algo])
    plt.errorbar(scores[:, 0], scores[:, 1], yerr=stds[:,1])
plt.legend(list(scores_per_algo.keys()))
plt.xlabel('n (log_2 # samples)')
plt.ylabel('neg-log-likelihood')
# plt.ylim(175, 175 + 80)
plt.grid()
plt.show()