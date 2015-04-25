import os
import pandas as pd
from pandas import DataFrame
import numpy as np


def collect_result_files(save=True):
    my_results = []
    netkit_results = []

    for dirpath, dnames, fnames in os.walk("../results/"):

        if dirpath == '../results/':
            continue

        for fname in fnames:
            df = pd.read_csv(os.path.join(dirpath, fname))
            df['dataset'] = dirpath.split('/')[-1]
            df['features'] = fname
            if fname == 'netkit':
                netkit_results.append(df)
            else:
                my_results.append(df)

    my_results = pd.concat(my_results)
    if len(netkit_results) is not 0:
        netkit_results = pd.concat(netkit_results)
    if save:
        my_results.to_csv('../results/my_results.csv')
        if len(netkit_results) is not 0:
            netkit_results.to_csv('../results/netkit_results.csv')
    return my_results, netkit_results


def agg_results(save=True):
    my_results, netkit_results = collect_result_files(save=True)
    my_results['args'] = my_results[['depth', 'n_cluster', 'restart']].sum(axis=1)
    my_results = my_results.drop(['depth', 'n_cluster', 'restart'])
    my_results.fillna(-1, inplace=True)
    grouped = my_results.groupby(['dataset', 'features', 'ratio', 'args'])
    my_agg_results = DataFrame({'mean_acc':grouped['cluster_acc'].mean(), 'std_acc':grouped['cluster_acc'].std()})

    if len(netkit_results) is not 0:
        df_wvrn = netkit_results[['wvrn_acc', 'wvrn_std', 'ratio', 'dataset']]
        df_wvrn.columns = ['mean_acc', 'std_acc', 'ratio', 'dataset']
        df_wvrn['features'] = 'wvrn'

        df_nlb = netkit_results[['nlb_acc', 'nlb_std', 'ratio', 'dataset']]
        df_nlb.columns = ['mean_acc', 'std_acc', 'ratio', 'dataset']
        df_nlb['features'] = 'nlb'

        df_wvrn_nlb = pd.concat([df_wvrn, df_nlb])
        df_wvrn_nlb['args'] = np.nan
        df_wvrn_nlb.set_index(['dataset', 'features', 'ratio', 'args'], inplace=True)

    if len(netkit_results) is not 0:
        df_all = pd.concat([df_wvrn_nlb, my_agg_results])
    else:
        df_all = my_agg_results

    if save:
        df_all.to_csv('../results/agg_results.csv')
    return df_all

agg_results(save=True)
my_results, netkit_results = collect_result_files(save=True)
