import matplotlib.pyplot as plt
from tcorex.experiments import data as data_tools, baselines_clustering
import numpy as np
import itertools
import os
import pickle
import argparse

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden', '-m', default=64, type=int,
                    help='number of hidden variables')
parser.add_argument('--p_exp_min', default=7, type=int,
                    help='exponent of min. number of observed variables')
parser.add_argument('--p_exp_max', default=15, type=int,
                    help='exponent of max. number of observed variables')
parser.add_argument('--snr', '-s', default=0.1, type=float,
                    help='signal-to-noise ratio')
parser.add_argument('--n_samples', '-n', default=300, type=int,
                    help='number of samples')
parser.add_argument('--n_runs', '-r', default=20, type=int,
                    help='number of experiments to run')
parser.add_argument('--num_extra_parents', default=0.1, type=float,
                    help='average number of extra parents for each x_i')
parser.add_argument('--num_correlated_zs', default=0, type=int,
                    help='number of zs each z_i is correlated with (besides z_i itself)')
parser.add_argument('--random_scale', dest='random_scale', action='store_true',
                    help='if true x_i will have random scales')
parser.add_argument('--device', '-d', type=str, default='cpu',
                    help='which device to use for pytorch corex')
parser.set_defaults(random_scale=False)
args = parser.parse_args()
print(args)

m = args.n_hidden
n = args.n_samples
p_min = args.p_exp_min
p_max = args.p_exp_max
n_runs = args.n_runs

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

        (baselines_clustering.ICA(name='ICA'), {
            'n_components': m
        }),

        (baselines_clustering.KMeans(name='k-Means'), {
            'n_clusters': m
        }),

        # (baselines_clustering.Spectral(name='Spec.'), {
        #     'n_clusters': m
        # }),

        (baselines_clustering.Hierarchical(name='Hier.'), {
            'n_clusters': m
        }),
    ]

file_path = f'scores_p{p_min}_{p_max}_m{m}.pkl'
if os.path.isfile(file_path):
    with open(file_path, 'rb') as f:
        scores_per_algo = pickle.load(f)
else:
    scores_per_algo = {}

r = list(range(p_min, p_max + 1))
n_experiments = len(r)
for k, i in enumerate(r):
    p = 2 ** i
    print("")
    print(f"Experiment {k+1} of {n_experiments}: p=2^{i}={p}")

    for j in range(0, n_runs):
        skip = True
        for (method, _) in methods:
            name = method.name
            if name not in scores_per_algo\
                    or str(i) not in scores_per_algo[name]\
                    or len(scores_per_algo[name][str(i)]) <= j:
                skip = False
                break

        if skip:
            print(f"Skipping run {j+1} of {n_runs} for p=2^{i}={p}")
            continue

        print("")
        print(f"Starting run {j+1} of {n_runs} for p=2^{i}={p}")

        data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=n, snr=args.snr,
                                                            num_extra_parents=args.num_extra_parents,
                                                            num_correlated_zs=args.num_correlated_zs,
                                                            random_scale=args.random_scale)
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

plt.figure(figsize=(6,3))
for algo, results in score_means.items():
    scores = np.array(results)
    stds = np.array(score_stds[algo])
    plt.errorbar(scores[:, 0], scores[:, 1], yerr=stds[:,1])
plt.legend(list(scores_per_algo.keys()))
plt.xlabel('p (log_2 # variables)')
plt.ylabel('Cluster score - ARI')
# plt.ylim(175, 175 + 80)
plt.grid()
plt.show()