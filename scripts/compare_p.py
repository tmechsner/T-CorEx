import matplotlib.pyplot as plt
from tcorex.experiments import data as data_tools, baselines_clustering
import numpy as np
import itertools
import os
import json

np.random.seed(42)

m = 64
n = 300
num_runs = 20

methods = [
        (baselines_clustering.PCA(name='PCA'), {
            'n_components': m
        }),

        (baselines_clustering.FactorAnalysis(name='Factor Analysis'), {
            'n_components': m
        }),

        (baselines_clustering.LinearCorex(name='Linear CorEx'), {
            'n_hidden': m,
            'max_iter': 500,
            'anneal': True
        }),
    ]

file_path = f'scores_p_m{m}.json'
if os.path.isfile(file_path):
    with open(file_path, 'r') as f:
        scores_per_algo = json.load(f)
else:
    scores_per_algo = {}

r = list(range(7, 16))
n_experiments = len(r)
for k, i in enumerate(r):
    p = 2 ** i
    print("")
    print(f"Experiment {k+1} of {n_experiments}: p=2^{i}={p}")

    for j in range(0, num_runs):
        skip = True
        for (method, _) in methods:
            name = method.name
            if name not in scores_per_algo\
                    or str(i) not in scores_per_algo[name]\
                    or len(scores_per_algo[name][str(i)]) <= j:
                skip = False
                break

        if skip:
            print(f"Skipping run {j+1} of {num_runs} for p=2^{i}={p}")
            continue

        print("")
        print(f"Starting run {j+1} of {num_runs} for p=2^{i}={p}")

        data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=n, snr=0.1,
                                                            num_extra_parents=0,
                                                            num_correlated_zs=0,
                                                            random_scale=False)
        print("")

        true_clusters = np.arange(m).repeat(p // m, axis=0)

        for (method, params) in methods:
            name = method.name
            if name in scores_per_algo\
                    and str(i) in scores_per_algo[name]\
                    and len(scores_per_algo[name][str(i)]) > j:
                print(f"Skipping {name}")
                continue
            else:
                print(f"Doing {name}")
            best_score, _, _, _ = method.select(data, true_clusters, params, verbose=False)
            if name not in scores_per_algo:
                scores_per_algo[name] = {}
            if i not in scores_per_algo[name]:
                scores_per_algo[name][i] = []
            scores_per_algo[name][i].append(best_score)

        with open(file_path, 'w') as f:
            json.dump(scores_per_algo, f)

with open(file_path, 'w') as f:
    json.dump(scores_per_algo, f)


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
plt.xlabel('p (log_2 # samples)')
plt.ylabel('Adjusted Rand Index')
# plt.ylim(175, 175 + 80)
plt.grid()
plt.show()