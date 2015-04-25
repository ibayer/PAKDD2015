import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import re
import subprocess
from sklearn.externals.joblib import Memory 
memory = Memory('cache/')


@memory.cache
def run(datapath, classifier='wvrn', inferencemethod='relaxlabel', verbose=True,
        cv_folds=10, pruneSingletons=True):
    print 'run ' + datapath
    """
    rclassifier: 
                    nolb-lr-distrib:  use the logistic regression link-based
                                        classifier where values are normalized.

                    wvrn:             use the weighted-vote Relational Neighbor
                                        classifier (previously referred to as
                                        pRN-the probabilistic Relational Neighbor)

                    nobayes:          use the network-only Bayes classifier (with a marko
                                        v random field formulation)
                    naivebayes:

    inferencemethod: 
                    relaxlabel, null, gibbs, iterative
    """
    if classifier == 'nLB':
        classifier = 'nolb-lr-distrib'

    from pandas import DataFrame
    sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    for sample in np.arange(0.1, 1, 0.1):

        commands_ = ['-rclassifier', classifier,
                    '-inferencemethod ' , inferencemethod, 
                     '-runs', str(cv_folds),
                     '-sample', str(sample),
                     '-stratified']
        if pruneSingletons:
            commands_.append('-pruneSingletons')
        commands_.append('data/NetKit-Data/' + datapath + '.arff')
        try:
            os.remove('cache/netkit-run-0.log')
        except OSError:
            pass

        path = os.path.join(os.getcwd(), 'netkit-srl-1.4.0/lib/NetKit-1.4.0.jar')
        output = subprocess.check_output(['java', '-jar', path] + commands_)

        try:
            pred = parse_output(output.split('\n'))
            log_data = parse_log('netkit-run-0.log')
            os.remove('netkit-run-0.log')
        except (OSError, KeyError) as e:
            print e
            print 'console output: ' + str(output)
            raise

        result = (log_data[0], log_data[1], sample)
        results.append(result)
        if verbose:
            print result
    return DataFrame.from_records(results, columns=['acc', 'std', 'ratio_labeled']), pred


def parse_output(output):
    """
        output: console output from netkit-srl
        returns: DataFrame with class probabilities
    """
    del output[-1]
    dic_list = []
    fold = ''
    for line in output:
        if '#' in line:
            fold = line.replace('#', '') 
        else:
            entries = line.split()
            pred_dic = {}
            for entry in entries[1:len(entries)]:
                tmp = entry.split(':')
                pred_dic[tmp[0]] = float(tmp[1])
            pred_dic['id'] = entries[0]
            pred_dic['fold'] = fold
            dic_list.append(pred_dic)

    return DataFrame.from_records(dic_list, index='id')


def parse_log(log_file='netkit-run-0.log'):
    """
        log_file: netkit-srl outputfile
        returns: (mean_accuracy, std)
    """
    fileObj = open(log_file)
    lines = fileObj.read()
    m = re.search('Accuracy-Final: 0.[0-9]*\s\(0.[0-9]*\)*', lines)
    res = m.group(0).split()
    return float(res[1]), float(res[2][1:len(res[2])-1])


if __name__ == '__main__':
    print 'hallo'
    import matplotlib.pyplot as plt
    wv_RN, pred  = run(classifier='wvrn')
    wv_RN
