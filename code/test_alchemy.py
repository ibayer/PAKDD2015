import scipy.sparse as sp
import numpy as np
from alchemy_experiments import bow_class_count_experiments


# only bow
def test_eval_bow():
    n_samples = 100
    X = np.zeros((n_samples,2))
    y = np.ones(n_samples)
    X[:n_samples/2, 1] = 1
    X[n_samples/2:, 0] = 1
    y[:n_samples/2] = 0
    print X, y
    n_runs = 1
    max_k=1
    results = bow_class_count_experiments(None, X, y, 'sklearn_class',
            max_k=max_k, n_runs=n_runs, test_run=True)
    print results


# only information in link
def test_eval_link():
    n_samples = 100
    X = np.zeros((n_samples,3))
    adj = np.zeros((n_samples, n_samples))
    y = np.ones(n_samples)
    # use block structure for adj matrix
    for i in range(n_samples):
        for j in range(n_samples):
            if i < n_samples/2 and j <= n_samples/2:
                adj[i,j] = 1
            if i >= n_samples/2 and j > n_samples/2:
                adj[i,j] = 1
# put useless stuff in X
    X[:,2] = 1
    y[:n_samples/2] = 0
    print X, y
    print adj
    n_runs = 1
    max_k=1
    adj = sp.csc_matrix(adj)
    results = bow_class_count_experiments(adj, X, y, 'sample_link',
            max_k=max_k, n_runs=n_runs, test_run=True)
    print results

#test_eval_bow()
test_eval_link()
