import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from experiments import evaluate_model
from read_files import read_linqs_data, read_netkit_data_in_graph
from feature_extract import create_metis_partitions


def ids_experiment(G, y, dataset, max_k, n_runs, test_run):
    from feature_extract import get_neighbor_ids

    adjs = get_neighbor_ids(G, y.index.values, max_k=max_k)

    results = []
    X = []

    for depth in range(max_k):
        X = sp.hstack(adjs[:depth+1])
        df = evaluate_model(X, y, n_runs=n_runs, test_run=test_run)
        df['depth'] = depth
        results.append(df)

    results = pd.concat(results)
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/ids', index=False)
    return results


def class_count_experiments(G, y, dataset, max_k, n_runs, test_run,
        ignore_unlabeled=False):
    from feature_extract import get_neighbor_ids

    adjs = get_neighbor_ids(G, y.index.values, max_k=max_k)

    results = []

    for depth in range(max_k):
        print "depth", depth
        df = evaluate_model(None, y, adjs=adjs[:depth+1], n_runs=n_runs,
                test_run=test_run,ignore_unlabeled=ignore_unlabeled)
        df['depth'] = depth
        results.append(df)

    results = pd.concat(results)
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    if ignore_unlabeled:
        results.to_csv('../results/' + dataset + '/ids_labeled', index=False)
    else:
        results.to_csv('../results/' + dataset + '/counts', index=False)
    return results


def class_prob_experiments(G, y, dataset, max_k, n_runs, test_run):
    from feature_extract import get_neighbor_ids

    adjs = get_neighbor_ids(G, y.index.values, max_k=max_k)

    results = []

    for depth in range(max_k):
        df = evaluate_model(None, y, adjs=adjs[:depth+1], normelize=True,
                n_runs=n_runs, test_run=test_run)
        df['depth'] = depth
        results.append(df)

    results = pd.concat(results)
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/proba', index=False)
    return results


def cluster_experiments(nets, y, dataset, n_runs, test_run):

    max_size = int(np.floor(np.log2(len(nets.nodes()))))
    sizes = [2**i for i in range(1, max_size + 1)]

    parts = create_metis_partitions([nets],sizes=sizes, seed=123)
    parts = parts[0]

    results = []

    ohe = OneHotEncoder(n_values='auto', categorical_features='all')

    y = y.loc[parts.index] # remove singletones

    for n_cluster in sizes:
        X = parts[str(n_cluster) + '_' + dataset].values + 1
        X = ohe.fit_transform(np.expand_dims(X, axis=1))
        df = evaluate_model(X, y, n_runs=n_runs, test_run=test_run)
        df['n_cluster'] = n_cluster
        results.append(df)

    # use all partiionings
    X = parts.values + 1
    X = ohe.fit_transform(X)
    df = evaluate_model(X, y, n_runs=n_runs, test_run=test_run)
    df['n_cluster'] = 0
    results.append(df)

    results = pd.concat(results)
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/cluster', index=False)
    return results


def rwr_experiments(G, y, dataset, n_runs, test_run):
    from feature_extract import get_neighbor_ids, calc_rwr_matrix

    adjs = get_neighbor_ids(G, y.index.values, max_k=1)

    results = []
    X = []
    c_range = np.arange(0.1, 1, 0.2)
    if test_run:
        c_range = [0.9]

    for c in c_range:
        X = calc_rwr_matrix(np.array(adjs[0].todense()), c)
        df = evaluate_model(X, y, n_runs=n_runs, test_run=test_run)
        df['restart'] = c
        results.append(df)

    results = pd.concat(results)
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/rwr', index=False)
    return results


def run_linqs_data_experiments(test_run=True):

    print 'linqs_data_experiments'
    n_runs = 10
    max_k=3
    if test_run:
        n_runs = 1
        max_k=2


    datasets = ['cora','citeseer', 'imdb_all']

    for dataset in datasets:
        print dataset
        if dataset == 'imdb_all':
            nets, _, y = read_netkit_data_in_graph('data/NetKit-Data/', [dataset])
            G = nets[0]
        else:
            G, X, y = read_linqs_data('data/LINQS/', dataset)

        non_singletones = [node for node, degree in G.degree().iteritems() if degree >0]
        y = y.loc[non_singletones] # remove singletones

        print 'cluster_experiments'
        results = cluster_experiments(G, y, dataset, n_runs=n_runs,
                test_run=test_run)

        print 'class_prob_experiments'
        results = class_prob_experiments(G, y, dataset, max_k=max_k, n_runs=n_runs,
                test_run=test_run)

        print 'class_count_experiments'
        results = class_count_experiments(G, y, dataset, max_k=max_k, n_runs=n_runs,
                test_run=test_run)

        print 'ids_experiments'
        results = ids_experiment(G, y, dataset, max_k=max_k, n_runs=n_runs,
                test_run=test_run)

        print 'rwr_experiments'
        results = rwr_experiments(G, y, dataset, n_runs=n_runs,
                test_run=test_run)

        print 'ignore unlabeled nodes experiments'
        results = class_count_experiments(G, y, dataset, max_k=max_k, n_runs=n_runs,
                test_run=test_run, ignore_unlabeled=True)


run_linqs_data_experiments(test_run=False)
