import pandas as pd
from pandas import DataFrame
import numpy as np
import networkx as nx


def read_netkit_data_in_graph(basedir,
        basenames=["industry-pr", "industry-yh"]):
    '''
    read netkit dataset from basedir
    expects subdirectories for each basename, a file "basename.rn" with relation and
    "basename.csv" with label data
    '''
    delimiter="[, ]"
    labels=dict()
    networks=[]
    for i in range(len(basenames)):
        basename=basenames[i]
        edges=pd.read_csv(basedir+basename+"/"+basename+".rn",names=["source","target","weight"],delimiter=delimiter)
        nodes=pd.read_csv(basedir+basename+"/"+basename+".csv",names=["index", "label"], delimiter=delimiter)
        g=nx.Graph()
        g.graph['name']=basename
        g.add_nodes_from(nodes['index'].values)
        g.add_weighted_edges_from(edges.values)
        g.remove_nodes_from([node for node,degree in g.degree().iteritems() if degree==0])
        networks.append(g)
        for n in nodes.values:
            labels[n[0]]=n[1]
        df_labels = DataFrame.from_dict(labels, orient='index')
        df_labels.columns = ['labels']
    return networks,[], df_labels


def read_linqs_data(basedir, basename):
    subpath = basename + '/' + basename
    df = pd.read_csv(basedir + subpath +'.content', delimiter='\t', header=None)
    df_labels = DataFrame(df.iloc[:,-1].values,
            index=df.iloc[:,0].values.astype(str), columns=['labels'])
    X = df.iloc[:,1:-1].values
    X = X.astype(np.float64)
    G = nx.read_edgelist(basedir + subpath + '.cites')
    G = G.to_undirected()
    G.graph['name']=basename
    return G, X, df_labels


def export_linqs_to_netkit():

    G, X, y = read_linqs_data('data/LINQS/', 'citeseer')
    no_labels = set(G.nodes()).difference(set(y.index.values))
    for node in no_labels:
        G.remove_node(node)
    nx.write_edgelist(G, 'data/NetKit-Data/citeseer/citeseer.rn',delimiter=',')

    G, X, y = read_linqs_data('data/LINQS/', 'cora')
    no_labels = set(G.nodes()).difference(set(y.index.values))
    for node in no_labels:
        G.remove_node(node)
    y.to_csv('data/NetKit-Data/cora/cora.csv')
    nx.write_edgelist(G, 'data/NetKit-Data/cora/cora.rn',delimiter=',')
