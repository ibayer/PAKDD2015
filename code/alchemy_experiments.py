import os
from alchemy import evaluate_model
import scipy.sparse as sp


def alchemy_experiments(G, X, y, dataset, max_k, n_runs, test_run):
    from feature_extract import get_neighbor_ids

    if G is not None:
        if sp.isspmatrix(G):
            adjs = G.tocsc()
        else:
            adjs = get_neighbor_ids(G, y.index.values, max_k=max_k)
    else:
        adjs = None

    results = []

    print 'bow + links'
    results = evaluate_model(sp.coo_matrix(X), y, adjs=adjs, n_runs=n_runs,
            test_run=test_run)
    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/alchemy_bow_link', index=False)

    return results


def run_alchemy_experiments(test_run=True):

    from read_files import read_linqs_data

    print 'run_bow_experiments'
    n_runs = 3
    max_k=1
    if test_run:
        n_runs = 1
        max_k=1

    data_dir = 'data/LINQS/'

    datasets = ['citeseer', 'cora']

    for dataset in datasets:
        print dataset
        G, X, y = read_linqs_data(data_dir, dataset)
        results = alchemy_experiments(G, X, y, dataset,
                max_k=max_k, n_runs=n_runs, test_run=test_run)


def test_alchemy_wrapper():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5,
            n_informative=5, n_repeated=0, n_redundant=0, n_classes=2)
    n_runs = 1
    max_k=1
    results = alchemy_experiments(None, X, y, 'sklearn_class',
            max_k=max_k, n_runs=n_runs, test_run=True)


run_alchemy_experiments(test_run=False)
