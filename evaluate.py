from __future__ import division
from __future__ import absolute_import

from generate_data import *
from misc_utils import make_sure_path_exists, make_buckets
from sklearn.model_selection import train_test_split
from theano_time_corex import *

import pickle
import argparse
import baselines
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nt', type=int, help='number of buckets')
    parser.add_argument('--nv', type=int, help='number of variables')
    parser.add_argument('--m', type=int, help='number of latent factors')
    parser.add_argument('--bs', type=int, help='block size')
    parser.add_argument('--train_cnt', default=16, type=int, help='number of train samples')
    parser.add_argument('--val_cnt', default=16, type=int, help='number of validation samples')
    parser.add_argument('--test_cnt', default=100, type=int, help='number of test samples')
    parser.add_argument('--snr', type=float, default=None, help='signal to noise ratio')
    parser.add_argument('--min_cor', type=float, default=0.8, help='minimum correlation between a child and parent')
    parser.add_argument('--max_cor', type=float, default=1.0, help='minimum correlation between a child and parent')
    parser.add_argument('--min_var', type=float, default=1.0, help='minimum x-variance')
    parser.add_argument('--max_var', type=float, default=1.0, help='maximum x-variance')
    parser.add_argument('--eval_iter', type=int, default=1, help='number of evaluation iterations')
    parser.add_argument('--prefix', type=str, default='', help='optional prefix of experiment name')
    parser.add_argument('--data_type', dest='data_type', action='store', default='syn_nglf_buckets',
                        choices=['syn_nglf_buckets', 'syn_general_buckets', 'syn_nglf_ts',
                                 'syn_general_ts', 'stock_day', 'stock_week', 'syn_nglf_buckets_smooth'],
                        help='which dataset to load/create')
    args = parser.parse_args()

    if args.data_type in ['syn_nglf_buckets', 'syn_nglf_ts']:
        args.nv = args.m * args.bs

    ''' Load data '''
    if args.data_type in ['syn_nglf_buckets', 'syn_general_buckets']:
        if args.data_type == 'syn_nglf_buckets':
            (data1, sigma1) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                       ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                       snr=args.snr, min_var=args.min_var, max_var=args.max_var,
                                                       min_cor=args.min_cor, max_cor=args.max_cor)
            (data2, sigma2) = generate_nglf_from_model(args.nv, args.m, args.nt // 2,
                                                       ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                       snr=args.snr, min_var=args.min_var, max_var=args.max_var,
                                                       min_cor=args.min_cor, max_cor=args.max_cor)
        else:
            (data1, sigma1) = generate_general_make_spd(args.nv, args.m, args.nt // 2,
                                                        ns=args.train_cnt + args.val_cnt + args.test_cnt)
            (data2, sigma2) = generate_general_make_spd(args.nv, args.m, args.nt // 2,
                                                        ns=args.train_cnt + args.val_cnt + args.test_cnt)

        data = data1 + data2
        args.ground_truth_covs = [sigma1 for i in range(args.nt // 2)] + [sigma2 for i in range(args.nt // 2)]
        args.train_data = [x[:args.train_cnt] for x in data]
        args.val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
        args.test_data = [x[-args.test_cnt:] for x in data]

    if args.data_type in ['syn_nglf_buckets_smooth']:
        (data, args.ground_truth_covs) = generate_nglf_smooth(nv=args.nv, m=args.m, nt=args.nt,
                                                              ns=args.train_cnt + args.val_cnt + args.test_cnt,
                                                              snr=args.snr, min_cor=args.min_cor, max_cor=args.max_cor,
                                                              min_var=args.min_var, max_var=args.max_var)
        args.train_data = [x[:args.train_cnt] for x in data]
        args.val_data = [x[args.train_cnt:args.train_cnt + args.val_cnt] for x in data]
        args.test_data = [x[-args.test_cnt:] for x in data]

    if args.data_type in ['syn_nglf_ts']:
        (data, args.ground_truth_covs) = generate_nglf_timeseries(
            nv=args.nv, m=args.m, nt=args.nt, ns=1+args.test_cnt, snr=args.snr,
            min_cor=args.min_cor, max_cor=args.max_cor,
            min_var=args.min_var, max_var=args.max_var)
        args.ts_data = data[:, 0, :]
        args.test_data = data[:, 1:, :]

    if args.data_type in ['stock_day', 'stock_week']:
        args.train_data, args.val_data, args.test_data = load_stock_data(
            nv=args.nv, train_cnt=args.train_cnt, val_cnt=args.val_cnt, test_cnt=args.test_cnt,
            data_type=args.data_type)
        args.ground_truth_covs = None
        args.nt = len(args.train_data)

    # some variables related to time-series case
    is_time_series = (args.data_type in ['syn_nglf_ts', 'syn_general_ts'])
    windows = [4, 8, 12]
    strides = ['one', 'half', 'full']

    ''' Define baselines and the grid of parameters '''
    methods = [
        (baselines.GroundTruth(name='Ground Truth',
                               covs=args.ground_truth_covs,
                               test_data=args.test_data), {}),

        (baselines.Diagonal(name='Diagonal'), {}),

        (baselines.LedoitWolf(name='Ledoit-Wolf'), {}),

        (baselines.OAS(name='Oracle approximating shrinkage'), {}),

        (baselines.PCA(name='PCA'), {'n_components': [args.m]}),

        (baselines.FactorAnalysis(name='Factor Analysis'), {'n_components': [args.m]}),

        (baselines.GraphLasso(name='Graphical LASSO (sklearn)'), {
            'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
            'mode': 'lars',
            'max_iter': 100}),

        (baselines.LinearCorex(name='Linear CorEx (applied bucket-wise)'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True}),

        (baselines.LinearCorexWholeData(name='Linear CorEx (applied on whole data)'), {
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True}),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO'), {
            'lamb': [0.01, 0.03, 0.1, 0.3],
            'beta': [0.03, 0.1, 0.3, 1.0],
            'indexOfPenalty': [1],  # TODO: extend grid of this one
            'max_iter': 100}),

        (baselines.TimeVaryingGraphLasso(name='T-GLASSO (no reg)'), {
            'lamb': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
            'beta': [0.0],
            'indexOfPenalty': [1],
            'max_iter': 100}),

        # (baselines.TCorex(tcorex=TCorex, name='T-Corex (Sigma)'), {
        #     'nv': args.nv,
        #     'n_hidden': args.m,
        #     'max_iter': 500,
        #     'anneal': True,
        #     'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #     'l2': [],
        #     'reg_type': 'Sigma'
        # }),

        (baselines.TCorex(tcorex=TCorex, name='T-Corex (W)'), {
            'nv': args.nv,
            'n_hidden': args.m,
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0, 0.001, 0.003],
            'l2': [],
            'reg_type': 'W'
        }),

        # (baselines.TCorex(tcorex=TCorex, name='T-Corex (MI)'), {
        #     'nv': args.nv,
        #     'n_hidden': args.m,
        #     'max_iter': 500,
        #     'anneal': True,
        #     'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #     'l2': [],
        #     'reg_type': 'MI'
        # }),

        # (baselines.TCorex(tcorex=TCorex, name='T-Corex (WWT)'), {
        #     'nv': args.nv,
        #     'n_hidden': args.m,
        #     'max_iter': 500,
        #     'anneal': True,
        #     'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        #     'l2': [],
        #     'reg_type': 'WWT'
        # }),

        (baselines.TCorex(tcorex=TCorexPrior1, name='T-Corex + priors (W, method 1)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0, 0.03, 0.1, 0.3, 1.0],
            'l2': [],
            # 'lamb': [0.0, 0.5, 0.9, 0.99],
            'lamb': [0.0],
            'reg_type': 'W',
            'init': True
        }),

        (baselines.TCorex(tcorex=TCorexPrior2, name='T-Corex + priors (W, method 2)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0, 0.03, 0.1, 0.3, 1.0],
            'l2': [],
            # 'lamb': [0.0, 0.5, 0.9, 0.99],
            'lamb': [0.0],
            'reg_type': 'W',
            'init': True
        }),

        (baselines.TCorex(tcorex=TCorexPrior2Weights, name='T-Corex + priors (W, method 2, weighted samples)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'l2': [],
            # 'lamb': [0.0, 0.5, 0.9, 0.99],
            'lamb': [0.5],
            # 'gamma': [1.25, 1.5, 2.0, 2.5, 1e5],
            'gamma': [1e5],
            'reg_type': 'W',
            'init': True
        }),

        (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, weighted samples)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'l2': [],
            # 'gamma': [1.25, 1.5, 2.0, 2.5, 1e5],
            'gamma': [1e5],
            'reg_type': 'W',
            'init': True
        }),

        (baselines.TCorex(tcorex=TCorexWeights, name='T-Corex (W, weighted samples, no init)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'l2': [],
            # 'gamma': [1.25, 1.5, 2.0, 2.5, 1e5],
            'gamma': [1e5],
            'reg_type': 'W',
            'init': False
        }),

        (baselines.TCorex(tcorex=TCorexWeightedObjective, name='T-Corex (W, weighted objective)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'l2': [],
            'gamma': [1.25, 1.5, 2.0, 2.5, 1e5],
            'reg_type': 'W',
            'init': True
        }),

        (baselines.TCorex(tcorex=TCorexWeightsMod, name='T-Corex (W, weighted samples, modified)'), {
            'nv': args.nv,
            'n_hidden': [args.m],
            'max_iter': 500,
            'anneal': True,
            # 'l1': [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            # 'l1': [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            'l1': [1.0, 3.0, 10.0, 30.0, 100.0],
            'l2': [],
            'gamma': [1.25, 1.5, 2.0, 2.5, 1e5],
            # 'gamma': [1e5],
            'reg_type': 'W',
            'init': True,
            'sample_cnt': 256
        })
    ]

    results = {}
    for (method, params) in methods[-1:]:
        name = method.name
        if not is_time_series:
            ''' Buckets '''
            best_params, best_score = method.select(args.train_data, args.val_data, params)
            results[name] = method.evaluate(args.train_data, args.test_data, best_params, args.eval_iter)
            results[name]['best_params'] = best_params
            results[name]['best_val_score'] = best_score
        else:
            ''' Time-series '''
            results_per_window_and_stride = []
            best_val_score = 1e18
            best_params = None
            best_window = None
            best_stride = None

            for window in windows:
                for stride in strides:
                    # use all static models with stride = 'minimum'
                    if name.lower().find('time') == -1 and stride != 'one':
                        continue

                    ''' Make bucketed data and split it into training and validation sets '''
                    data, test_data = make_buckets(args.ts_data, args.test_data, window, stride)
                    if len(data) == 1:  # the window size is too big
                        continue

                    train_data = []
                    val_data = []
                    for t in range(len(data)):
                        cur_train, cur_val = train_test_split(data[t], test_size=max(1, int(0.15 * len(data[t]))))
                        train_data.append(cur_train)
                        val_data.append(cur_val)

                    ''' Select hyper-parameters other than window and stride '''
                    cur_best_params, cur_best_val_score = method.select(train_data, val_data, params)
                    if best_params is None or cur_best_val_score < best_val_score:
                        best_params = cur_best_params
                        best_val_score = cur_best_val_score
                        best_window = window
                        best_stride = stride

                    ''' Evaluate on the test set '''
                    test_scores = method.evaluate(train_data, test_data, cur_best_params, args.eval_iter)
                    test_scores['window'] = window
                    test_scores['stride'] = stride
                    test_scores['cur_best_val_score'] = cur_best_val_score
                    test_scores['cur_best_params'] = cur_best_params
                    results_per_window_and_stride.append(test_scores)

            train_data, test_data = make_buckets(args.ts_data, args.test_data, best_window, best_stride)
            results[name] = method.evaluate(train_data, test_data, best_params, args.eval_iter)
            results[name]['best_params'] = best_params
            results[name]['best_window'] = best_window
            results[name]['best_stride'] = best_stride
            results[name]['results_per_window_and_stride'] = results_per_window_and_stride

    ''' Save the data and results '''
    print("Saving the data and parameters of the experiment ...")

    if args.data_type in ['syn_nglf_buckets', 'syn_general_buckets', 'syn_nglf_ts', 'syn_general_ts']:
        exp_name = '{}.nt{}.m{}.bs{}.train_cnt{}.val_cnt{}.test_cnt{}'.format(
            args.data_type, args.nt, args.m, args.bs, args.train_cnt, args.val_cnt, args.test_cnt)
        if args.snr:
            suffix = '.snr{:.2f}'.format(args.snr)
        else:
            suffix = '.min_cor{:.2f}.max_cor{:.2f}'.format(args.min_cor, args.max_cor)
        exp_name = exp_name + suffix
    if args.data_type in ['stock_day', 'stock_week']:
        exp_name = '{}.m{}.train_cnt{}.val_cnt{}.test_cnt{}'.format(
            args.data_type, args.m, args.train_cnt, args.val_cnt, args.test_cnt)

    if args.prefix != '':
        exp_name = args.prefix + '.' + exp_name

    results_path = "results/{}.results.json".format(exp_name)
    print("Saving the results in {}".format(results_path))
    make_sure_path_exists(results_path)
    with open(results_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
