from read_files import read_linqs_data
from collections import Counter
import networkx as nx

print ' ----  LINQS version ------ '
data_dir = 'data/LINQS/'
datasets = ['citeseer', 'cora']

for dataset in datasets:
    G, X, y = read_linqs_data(data_dir, dataset)
    print dataset
    print len(y), 'samples', len(set(y['labels'])), 'classes'
    count = Counter(y['labels']).most_common(1)
    print 'most frequent', count
    print 'base accuracy', Counter(y['labels']
                                   ).most_common(1)[0][1] / float(len(y))
    print 'bow size', X.shape
    print nx.info(G)

from read_files import read_netkit_data_in_graph
print ' ----  netkit version ------ '

data_dir = 'data/NetKit-Data/'
datasets = ['cora', 'citeseer', "imdb_all"]

for dataset in datasets:
    nets, _, y = read_netkit_data_in_graph(data_dir, [dataset])
    G = nets[0]
    print dataset
    print len(y), 'samples', len(set(y['labels'])), 'classes'
    count = Counter(y['labels']).most_common(1)
    print 'most frequent', count
    print 'base accuracy', Counter(y['labels']
                                   ).most_common(1)[0][1] / float(len(y))
    print nx.info(G)
    print '-- without singletones --'
    non_singletones = [node for node, degree in G.degree().iteritems() if
                       degree > 0]
    y = y.loc[non_singletones]  # remove singletones
    print len(y), 'samples', len(set(y['labels'])), 'classes'
    count = Counter(y['labels']).most_common(1)
    print 'most frequent', count

    print 'base accuracy', Counter(y['labels']
                                   ).most_common(1)[0][1] / float(len(y))

    print
