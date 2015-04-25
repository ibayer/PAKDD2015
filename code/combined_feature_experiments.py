import os
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from experiments import evaluate_model
from read_files import read_netkit_data_in_graph
from feature_extract import create_metis_partitions


def id_class_count_experiments(G, y, dataset, max_k, n_runs, test_run):
    from feature_extract import get_neighbor_ids

    adjs = get_neighbor_ids(G, y.index.values, max_k=max_k)

    results = []
    X = adjs[0].todense()

    results = evaluate_model(X, y, adjs=adjs, n_runs=n_runs,
            test_run=test_run)

    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/id_counts', index=False)
    return results


def id_class_proba_experiments(G, y, dataset, max_k, n_runs, test_run):
    from feature_extract import get_neighbor_ids

    adjs = get_neighbor_ids(G, y.index.values, max_k=max_k)

    results = []
    X = adjs[0].todense()

    results = evaluate_model(X, y, adjs=adjs, normelize=True, n_runs=n_runs,
            test_run=test_run)

    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/id_proba', index=False)
    return results


def id_cluster_experiments(nets, y, dataset, n_runs, test_run):
    from feature_extract import get_neighbor_ids

    adjs = get_neighbor_ids(nets, y.index.values, max_k=1)

    max_size = int(np.floor(np.log2(len(nets.nodes()))))
    sizes = [2**i for i in range(1, max_size + 1)]

    parts = create_metis_partitions([nets],sizes=sizes, seed=123)
    parts = parts[0]

    results = []

    ohe = OneHotEncoder(n_values='auto', categorical_features='all')

    y = y.loc[parts.index] # remove singletones


    # use all partiionings
    X = parts.values + 1
    X = ohe.fit_transform(X)
    X = sp.hstack([X, adjs[0]])
    results = evaluate_model(X, y, n_runs=n_runs, test_run=test_run)

    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/id_cluster', index=False)
    return results


def id_rwr_experiments(G, y, dataset, n_runs, test_run):
    from feature_extract import get_neighbor_ids, calc_rwr_matrix

    adjs = get_neighbor_ids(G, y.index.values, max_k=1)

    X = []

    X = calc_rwr_matrix(np.array(adjs[0].todense()), 0.9)
    X = np.hstack([X, adjs[0].todense()])
    results = evaluate_model(X, y, n_runs=n_runs, test_run=test_run)

    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/id_rwr', index=False)
    return results


def run_combined_features_experiments(test_run=True):

    from read_files import read_linqs_data
    n_runs = 10
    max_k=3
    if test_run:
        n_runs = 1
        max_k=2

    datasets = ['imdb_all', 'cora','citeseer']

    for dataset in datasets:
        print dataset
        if dataset == 'imdb_all':
            nets, _, y = read_netkit_data_in_graph('data/NetKit-Data/', [dataset])
            G = nets[0]
        else:
            G, X, y = read_linqs_data('data/LINQS/', dataset)

        non_singletones = [node for node, degree in G.degree().iteritems() if degree >0]
        y = y.loc[non_singletones] # remove singletones

        results = id_cluster_experiments(G, y, dataset, n_runs=n_runs,
                test_run=test_run)

        results = id_class_proba_experiments(G, y, dataset, max_k=max_k, n_runs=n_runs,
                test_run=test_run)
        results = id_class_count_experiments(G, y, dataset, max_k=max_k, n_runs=n_runs,
                test_run=test_run)
        results = id_rwr_experiments(G, y, dataset, n_runs=n_runs,
                test_run=test_run)

run_combined_features_experiments(test_run=False)
