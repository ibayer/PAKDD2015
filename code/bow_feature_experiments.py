import os
from experiments import evaluate_model
from read_files import read_linqs_data


def bow_class_count_experiments(G, X, y, dataset, max_k, n_runs, test_run):
    from feature_extract import get_neighbor_ids

    adjs = get_neighbor_ids(G, y.index.values, max_k=max_k)

    results = []

    # bow and counts
    print 'bow + counts'
    results = evaluate_model(X, y, adjs=adjs, n_runs=n_runs,
            test_run=test_run)

    if not os.path.exists('../results/' + dataset + '/'):
        os.makedirs('../results/' + dataset + '/')
    results.to_csv('../results/' + dataset + '/bow_counts', index=False)

    # bow
    print 'bow'
    results = evaluate_model(X, y, adjs=None, n_runs=n_runs,
            test_run=test_run)
    results.to_csv('../results/' + dataset + '/bow', index=False)

    # counts
    print 'counts'
    results = evaluate_model(None, y, adjs=adjs, n_runs=n_runs,
            test_run=test_run)
    results.to_csv('../results/' + dataset + '/counts', index=False)
    return results


def run_bow_class_count_experiments(test_run=True):

    print 'run_bow_experiments'
    n_runs = 10
    max_k=3
    if test_run:
        n_runs = 1
        max_k=2

    data_dir = 'data/LINQS/'

    datasets = ['citeseer', 'cora']

    for dataset in datasets:
        print dataset
        G, X, y = read_linqs_data(data_dir, dataset)
        results = bow_class_count_experiments(G, X, y, dataset,
                max_k=max_k, n_runs=n_runs, test_run=test_run)


run_bow_class_count_experiments(test_run=False)
