import networkx as nx
from scipy.sparse import lil_matrix, coo_matrix
import numpy as np
import scipy.linalg as LA
from pandas import DataFrame
from sklearn.preprocessing import normalize

def get_neighbor_ids(g, index, max_k=3):
    '''
    extract feature matrices for the node neighborhood in g

    parameters
    ==========

    g the networkx network to encode

    index list of all nodes in g (as given by g.nodes()) providing the row/column order of the resulting features

    max_k maximal distance to encode

    return
    ======

    list of matrices corrospondig to distances 1..max_k with [na,nb]=1 if distance(na,nb)=d for d'th matrix in list

    '''
    rev_index=dict([(index[i],i) for i in range(len(index))])
    l=nx.all_pairs_shortest_path_length(g,cutoff=max_k)
    features=[lil_matrix((len(index),len(index))) for _ in range(max_k)]
    for node in index:
        row=rev_index[node]
        if node in l:
            for neighbor,distance in l[node].iteritems():
                if distance > 0 and neighbor in index:
                    features[distance-1][row,rev_index[neighbor]]=1
    return features


def setup_label_matrix(y, n_classes=None):

        if n_classes is None:
            n_classes = len(set(y))
            # -1 is not a clas label
            if -1 in set(y):
                n_classes -= 1
        n_samples = len(y)

        row = np.where(y >= 0)[0]
        col = y[y >= 0]
        data = np.ones(col.shape)
        C = coo_matrix((data, (row, col)),
                shape=(n_samples, n_classes)).todense()
        C[np.array(C.sum(axis=1)).flatten()
                < 1, :] = np.nan
        return C


def calc_rwr_matrix(A, c=0.9):

    A_norm = normalize(A, norm='l1', axis=1, copy=True)
    Q = np.identity(A_norm.shape[0], dtype=float) - c * A_norm
    rwr = (1 - c) * LA.inv(Q)
    normalize(rwr, norm='l2',
            axis=0, copy=False)
    return rwr


def create_metis_partitions(networks, sizes=[2,4,8,16,32,64,128,256], use_weights=False, seed=666):
    '''partition all networks and return unified result (partition tables for joint node set)

    Parameters
    ----------
    networks :  array of networkx Graphs
    sizes : list of partitioning sizes
         numbers of partitions to split the graphs into
    use_weights : (optional) boolean
         will use the 'weight' attribute of the networks edges for partitioning (True) or
         partition the unweighted graphs (False)
    seed : (optional) integer
         used as seed for the metis partitioning (for each call the same seed)
    '''
    p = []
    import os
    os.environ['METIS_DLL'] = "/usr/local/lib/libmetis.so"
    import metis
    for g in networks:
        partitions = []
        if use_weights:
            g.graph['edge_weight_attr']='weight'
        for i in sizes:
            (_, parts) = metis.part_graph(g,i, seed=seed)
            partitions.append(parts)
        df = DataFrame(partitions).transpose()
        df.index = g.nodes()
        df.columns = [str(i)+'_'+g.graph['name'] for i in sizes]
        p.append(df)
        if use_weights:
            del g.graph['edge_weight_attr']
    return p
