import numpy as np
import scipy.sparse as sp
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, normalize
from pandas import DataFrame

from feature_extract import setup_label_matrix


def evaluate_model(X, y, adjs=None, normelize=False, n_runs=2, verbose=False,
                   test_run=False, ignore_unlabeled=False):

    le = LabelEncoder()
    # always copy to avoid side effects
    y = y.copy()

    y = y['labels'].values
    if X is not None and sp.isspmatrix(X):
        X = X.tocsc()

    # transform labels to integers
    y = le.fit_transform(y)

    logreg = LogisticRegression(penalty='l2', C=.5)

    from sklearn.grid_search import GridSearchCV
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'penalty': ['l2']}

    if test_run:
        parameters = {'C': [10], 'penalty': ['l2']}
    clf = GridSearchCV(logreg, parameters, n_jobs=-1, cv=4)

    results = []
    random_seeds = np.arange(n_runs)
    for run, seed in enumerate(random_seeds):
        for r in np.arange(0.1, 1, 0.1):
            i_train, i_test, y_train, y_test = \
                train_test_split(np.arange(len(y)), y, test_size=1-r,
                                 random_state=seed)

            # calc neighbor probs / counts
            if adjs is not None:
                y_train_semi = y.copy()
                # remove the unknown labels
                y_train_semi[i_test] = -1
                n_classes = len(set(y))
                label_matrix = setup_label_matrix(y_train_semi, n_classes)
                label_matrix[np.isnan(label_matrix)] = 0

                neigbhor_probs = []

                if isinstance(adjs, list):
                    for adj in adjs:
                        adj = adj.tocsc()
                        tmp = adj.dot(label_matrix)
                        if normelize:
                            normalize(tmp, norm='l1', axis=1, copy=False)
                        if ignore_unlabeled:
                            tmp = adj.todense()
                            tmp[:, i_test] = 0
                        neigbhor_probs.append(tmp)

                    X_count = np.hstack(neigbhor_probs)
                else:
                    adjs = adjs.tocsc()
                    tmp = adj.dot(label_matrix)
                    if normelize:
                        normalize(tmp, norm='l1', axis=1, copy=False)
                    if ignore_unlabeled:
                        tmp = adj.todense()
                        tmp[:, i_test] = 0
                    X_count = tmp

                # add other attributes
                if X is not None:
                    X_ext = np.hstack([X_count, X])
                else:
                    X_ext = X_count
            else:
                X_ext = X

            clf.fit(X_ext[i_train, :], y_train)
            print clf.best_params_
            cluster_acc = accuracy_score(clf.predict(X_ext[i_test, :]), y_test)
            cluster_acc_train = accuracy_score(clf.predict(X_ext[i_train, :]),
                                               y_train)
            if verbose:
                print 'ratio', r
                print cluster_acc

            results.append((r, run, cluster_acc, cluster_acc_train,
                            clf.best_params_['C']))

    df = DataFrame.from_records(results)
    df.columns = ['ratio', 'run',
                  'cluster_acc', 'cluster_acc_train', 'best_C']

    if verbose:
        print df.head(9)
    return df
