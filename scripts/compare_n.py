import matplotlib.pyplot as plt
from tcorex.experiments import data as data_tools, baselines_cov_est
from tcorex.experiments.misc import make_sure_path_exists
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import os
import json


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
    (baselines_cov_est.LVGLASSO(name='LVGLASSO'), {})
]

file_path = f'scores_n_m{m}.json'
if os.path.isfile(file_path):
    with open(file_path, 'r') as f:
        scores_per_algo = json.load(f)
else:
    scores_per_algo = {}
    for i in range(3, 8):
        n = 2 ** i
        with open(f'outputs/blessing/best/blessing_experiment_n{n}_p{p}_m{m}.results.json', 'r') as f:
            results = json.load(f)

        for j in range(0, 5):

            data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=n + 1000, snr=0.1,
                                                                num_extra_parents=0,
                                                                num_correlated_zs=0,
                                                                random_scale=False)

            train_data, test_data = train_test_split(data, test_size=1000, shuffle=True)

            for (method, params) in methods:
                try:
                    name = method.name
                    best_score, _, _, _, _, best_ari = method.select([train_data], [test_data], results[name]['best_params'], verbose=False)
                    if name not in scores_per_algo:
                        scores_per_algo[name] = []
                    scores_per_algo[name].append([i, best_score])
                except Exception:
                    pass

    with open(file_path, 'w') as f:
        json.dump(scores_per_algo, f)


scores_per_algo = {algo: {k: np.array([x[1] for x in v]) for k, v in itertools.groupby(results, key=lambda entry: entry[0])}
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