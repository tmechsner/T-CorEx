from tcorex.experiments import data as data_tools, baselines_cov_est
from tcorex.experiments.misc import make_sure_path_exists
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import pickle
import os
import json


def main(n=300, m=64, p=128):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hidden', '-m', default=m, type=int,
                        help='number of hidden variables')
    parser.add_argument('--n_observed', '-p', default=p, type=int,
                        help='number of observed variables')
    parser.add_argument('--snr', '-s', default=0.1, type=float,
                        help='signal-to-noise ratio')
    parser.add_argument('--n_samples', '-n', default=n, type=int,
                        help='number of samples')
    parser.add_argument('--num_extra_parents', default=0.1, type=float,
                        help='average number of extra parents for each x_i')
    parser.add_argument('--num_correlated_zs', default=0, type=int,
                        help='number of zs each z_i is correlated with (besides z_i itself)')
    parser.add_argument('--random_scale', dest='random_scale', action='store_true',
                        help='if true x_i will have random scales')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                        help='which device to use for pytorch corex')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/blessing/')
    parser.add_argument('--left', type=int, default=0)
    parser.add_argument('--right', type=int, default=-1)
    parser.set_defaults(random_scale=False)
    args = parser.parse_args()
    print(args)

    p = args.n_observed
    m = args.n_hidden
    snr = args.snr
    n = args.n_samples
    assert p % m == 0

    # generate some data
    data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=n, snr=snr,
                                                        num_extra_parents=args.num_extra_parents,
                                                        num_correlated_zs=args.num_correlated_zs,
                                                        random_scale=args.random_scale)

    true_clusters = np.arange(m).repeat(p // m, axis=0)

    train_data, tmp = train_test_split(data, test_size=0.4, shuffle=True)
    test_data, val_data = train_test_split(tmp, test_size=0.5, shuffle=False)

    data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=n, snr=snr,
                                                        num_extra_parents=args.num_extra_parents,
                                                        num_correlated_zs=args.num_correlated_zs,
                                                        random_scale=args.random_scale)

    eval_data, _ = data_tools.generate_approximately_modular(nv=p, m=m, ns=1000, snr=snr,
                                                             num_extra_parents=args.num_extra_parents,
                                                             num_correlated_zs=args.num_correlated_zs,
                                                             random_scale=args.random_scale)


    methods = [
        (baselines_cov_est.Diagonal(name='Diagonal'), {}),

        (baselines_cov_est.LedoitWolf(name='Ledoit-Wolf'), {}),

        (baselines_cov_est.PCA(name='PCA'), {
            'n_components': m
        }),

        (baselines_cov_est.SparsePCA(name='SparsePCA'), {
            'n_components': m,
            'alpha': [0.1, 0.3, 1.0, 3.0, 10.0],
            'ridge_alpha': [0.01],
            'tol': 1e-3,
            'max_iter': 100,  # NOTE: tried 500 no improvement, just slows down a lot !
        }),

        (baselines_cov_est.FactorAnalysis(name='Factor Analysis'), {
            'n_components': m
        }),

        (baselines_cov_est.GraphLasso(name='Graphical LASSO (sklearn)'), {
            'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'mode': 'lars',
            'max_iter': 500,
        }),

        (baselines_cov_est.LinearCorex(name='Linear CorEx'), {
            'n_hidden': m,
            'max_iter': 500,
            'anneal': True
        }),

        (baselines_cov_est.LVGLASSO(name='LVGLASSO'), {
            'alpha': [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
            'tau': [0.01, 0.1, 1.0, 10.0, 100.0],
            'rho': 1.0 / np.sqrt(train_data.shape[0]),  # NOTE works good, also rho doesn't change much
            'max_iter': 500,  # NOTE: tried 1000 no improvement
            'verbose': False
        })
    ]

    exp_name = f'blessing_experiment_n{n}_p{p}_m{m}'

    best_results_path = "{}.results.json".format(exp_name)
    best_results_path = os.path.join(args.output_dir, 'best', best_results_path)
    make_sure_path_exists(best_results_path)

    all_results_path = "{}.results.json".format(exp_name)
    all_results_path = os.path.join(args.output_dir, 'all', all_results_path)
    make_sure_path_exists(all_results_path)

    best_results = {}
    all_results = {}

    # read previously stored values
    if os.path.exists(best_results_path):
        with open(best_results_path, 'r') as f:
            best_results = json.load(f)
    if os.path.exists(all_results_path):
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)

    for (method, params) in methods:  #   [args.left:args.right]:
        name = method.name
        try:
            if name not in best_results:
                best_score, best_params, _, _, all_cur_results = method.select([train_data], [val_data], params)

                best_results[name] = {}
                best_results[name]['test_score'] = method.evaluate([test_data], best_params)
                best_results[name]['best_params'] = best_params
                best_results[name]['best_val_score'] = best_score

                all_results[name] = all_cur_results

                with open(best_results_path, 'w') as f:
                    json.dump(best_results, f)

                with open(all_results_path, 'w') as f:
                    json.dump(all_results, f)

                print("Best results are saved in {}".format(best_results_path))
                print("All results are saved in {}".format(all_results_path))
            else:
                # print(f"Skipping {name}, as there are results already")
                best_score, _, _, _, _ = method.select([eval_data], [eval_data], best_results[name]['best_params'], verbose=False)
                print(f"{name} eval: {best_score}")
        except Exception as e:
            print(str(e))

        # if args.method == 'linearcorex':
        #     pred_clusters = method.mis.argmax(axis=0)
        # else:
        #     pred_clusters = method.clusters()
        #
        # score = adjusted_rand_score(labels_true=true_clusters, labels_pred=pred_clusters)
        # print(pred_clusters, score)


if __name__ == '__main__':
    for i in range(3, 12):
        main(n=2**i, m=8, p=128)
        main(n=2**i, m=32, p=128)
