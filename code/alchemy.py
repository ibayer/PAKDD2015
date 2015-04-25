import scipy.sparse as sp
import numpy as np
import subprocess
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def evaluate_model(X, y, adjs=None, normelize=False, n_runs=2, verbose=False,
        test_run=False, ignore_unlabeled=False):

    if isinstance(adjs, list):
        adjs = adjs[0]

    le = LabelEncoder()
    # always copy to avoid side effects
    y = y.copy()

    if not isinstance(y, np.ndarray):
        y = y.labels.values
    if X is not None and sp.isspmatrix(X):
        X = X.tocsc()

    # transform labels to integers
    y = le.fit_transform(y)
    n_class = len(set(y))

    results = []
    random_seeds = np.arange(n_runs)
    for run, seed in enumerate(random_seeds):
        print "run", run
        for r in np.arange(0.1, 1, 0.1):

            print "ratio", r
            from time import gmtime, strftime
            print strftime("%Y-%m-%d %H:%M:%S", gmtime())

            i_train, i_test, y_train, y_test = \
                    train_test_split(np.arange(len(y)), y, test_size=1-r, random_state=seed)

            cluster_acc_train = np.nan
            cluster_acc = np.nan
            try:
                y_pred = run_alchemy(X, i_train, i_test, y, n_class, X_rel=adjs)
                y_pred_map = y_pred.argmax(axis=1)
                cluster_acc = accuracy_score(y_pred_map[i_test], y_test)
                cluster_acc_train = accuracy_score(y_pred_map[i_train], y_train)
                print "acc", cluster_acc, "acc train", cluster_acc_train
            except OSError:
                print ('failed to fit model')
            if verbose:
                print 'ratio', r
                print cluster_acc

            results.append((r, run,
                        cluster_acc, cluster_acc_train, np.nan))

    df = DataFrame.from_records(results)
    df.columns = ['ratio', 'run',
                'cluster_acc','cluster_acc_train', 'best_C']

    if verbose:
        print df.head(9)
    return df


def run_alchemy(X, i_train, i_test, y, n_class, X_rel=None):

    n_samples = X.shape[0]

    import os
    folder = "/tmp/" + str(os.getpid()) + "_"
    if X_rel == None:
        model_mln = "alchemy-2/datasets/tutorial/text-class/text-class.mln"
        train_data_db = folder + "text-class-train.db"
    else:
        model_mln = "alchemy-2/datasets/tutorial/text-class/hypertext-class.mln"
        train_data_db = folder + "text-class-train.db," + folder +"links-train.db"
    link_db = folder +"links-train.db"
    model_out_mln = folder + "text-class-out.mln"
    test_data_db = folder + "text-class-test.db"
    test_results = folder + "text-class.results"

    alchemy_dump(X, i_train, i_test, y, X_rel,
            folder + "text-class-train.db", test_data_db, link_db)

    learn_commands = ['alchemy-2/bin/learnwts', '-d',
                '-i', model_mln,
                '-o', model_out_mln,
                '-t', train_data_db,
                '-ne', 'Topic']

    infer_commands = ['alchemy-2/bin/infer',
                '-ms',
                '-i', model_out_mln,
                '-r', test_results,
                '-e', test_data_db,
                '-q',
                'Topic']
    try:
        output = subprocess.check_output(learn_commands)
    except OSError:
        print ('failed to fit model')

    try:
        output = subprocess.check_output(infer_commands)
    except OSError:
        print ('failed to predict')
    return parse_results(n_samples, n_class, test_results)


def parse_results(n_samples, n_class, i_file="/tmp/text-class.results"):
    y_pred = np.empty((n_samples, n_class))
    y_pred[:, :] = np.NAN

    def parse_line(line):
        value = float(line.split(" ")[1])
        row = int(line.split('"p')[1].split('"')[0])
        label = int(line.split('(C')[1].split(',')[0])
        y_pred[row, label] = value

    with open(i_file) as f:
        for line in f:
            parse_line(line)
    return y_pred


def alchemy_dump(X, i_train, i_test, y, X_rel=None,
                 o_train="/tmp/text-class-train.db", o_test="/tmp/text-class-test.db",
                 o_links="/tmp/links-train.db"):

    with open(o_train, "w") as output_train, open(o_test, "w") as output_test, \
        open(o_links, "w") as output_link:

        #training
        if X_rel is not None:
            dump_link(output_link, X_rel)
        dump_text(output_train, X, i_train, y)
        # predict
        dump_text(output_test, X, i_test)
        if X_rel is not None:
            dump_link(output_test, X_rel)
        print >> output_train
        print >> output_test
        print >> output_link


def dump_text(o_file, X, i_rows, y=None, offset=0):
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    A = sp.coo_matrix(X.tocsr())

    for i,j,v in zip(A.row, A.col, A.data):
        if i in i_rows:
            print >> o_file, 'HasWord("w' + str(j) + '","p' + str(i) + '")'

    if y is not None:
        for i, val in enumerate(y):
            if i in i_rows:
                print >> o_file, 'Topic(C' + str(val) + ',"p' + str(i) + '")'


def dump_link(o_file, X_rel):
    A = sp.coo_matrix(X_rel.tocsr())

    for i,j,v in zip(A.row, A.col, A.data):
        print >> o_file, 'Links("p' + str(j) + '","p' + str(i) + '")'


if __name__ == '__main__':
    n_samples = 4
    n_words = 8
    i_train = [0,1]
    i_test = [2, 3]
    n_class = 2

    np.random.seed(123)

    y = np.random.randint(0, 2, n_samples)

    X_bow = sp.rand(n_samples, n_words, .5)
    X_bow.data[X_bow.data >0] = 1
    X_rel = sp.rand(n_samples, n_samples, .5)
    X_rel.data[X_rel.data >0] = 1
    print 'X_bow\n', X_bow.todense()
    print 'X_rel\n', X_rel.todense()
    print 'y', y
    print '------- train / test split ----------'
    X_train_bow = X_bow.tolil()[i_train,:]
    X_test_bow = X_bow.tolil()[i_test,:]
    y_train = y[i_train]
    y_test = y[i_test]

    print 'X_train_bow \n', X_train_bow.todense()
    print 'X_test_bow \n', X_test_bow.todense()
    print 'y_train', y_train, 'y_test', y_test
