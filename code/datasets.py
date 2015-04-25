import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import thread as thr
from numpy.matlib import eye
from numpy.random import randint, multivariate_normal, rand
from networkx.generators.geometric import random_geometric_graph
from pandas import DataFrame


class Blockgraph:
    def create_blocked_relation(self, seed):
        n = self.nsamples
        g = self.nsamples / self.groups
        s = seed
        a = nx.to_numpy_matrix(nx.gnp_random_graph(
            n, float(self.inter_degree) / n, seed=s, directed=False))
        s = s + 1
        for i in range(self.groups):
            ga = nx.to_numpy_matrix(
                nx.gnp_random_graph(g, float(self.intra_degree) /
                                    g, s, directed=False))
            s = s + 1
            a[(i * g):((i + 1) * g), (i * g):((i + 1) * g)] = ga
        return a

    def create_features(self, dim_informative, dim_uninformative):
        features = np.zeros(shape=(self.nsamples, 0))
#         f=make_classification(self.nsamples,
        #  n_features, n_informative, n_redundant, n_repeated,
        #  n_classes=self.groups, n_clusters_per_class=1, weights,
        #  flip_y, class_sep, hypercube, shift, scale, shuffle, random_state)
        return features

    def __init__(self, samples=400, groups=2, intra_degree=10, inter_degree=2,
                 true_rel=1, wrong_rel=1, seed=33):
        self.nsamples = samples
        self.groups = groups
        # links with group
        self.intra_degree = intra_degree
        # links between groups
        self.inter_degree = inter_degree
        self.seed = seed
        self.labels = []
        for g in range(self.groups):
            self.labels.extend([g] * (self.nsamples / self.groups))
        s = seed
        self.relations = []
        for _ in range(true_rel):
            self.relations.append(self.create_blocked_relation(s))
            s = s + 1
        for _ in range(wrong_rel):
            p = np.random.permutation(samples)
            self.relations.append(self.create_blocked_relation(s)[p, :][:, p])
            s = s + 1
        self.features = self.create_features(1, 1)


def create_block_graph(n_samples=400, n_groups=2, intra_degree=5,
                       inter_degree=2, n_true_rel=1, n_wrong_rel=1, seed=3):
    b = Blockgraph(samples=n_samples, groups=n_groups,
                   intra_degree=intra_degree, inter_degree=inter_degree,
                   true_rel=n_true_rel, wrong_rel=n_wrong_rel, seed=seed)
    return [np.array(r) for r in b.relations], b.features, b.labels


class multi_gauss:
    def __init__(self, n_clusters=4, dim=2):
        self.centers = np.array([[2 * rand() for _ in range(dim)
                                  ] for _ in range(n_clusters)])
        self.cov = [eye(dim) * 0.2 * rand() for _ in range(n_clusters)]
        self.clusters = n_clusters

    def sample(self, n=1):
        c = randint(0, self.clusters - 1, size=n)
        if n == 1:
            return multivariate_normal(self.centers[c[0]],
                                       self.cov[c[0]], 1)[0]
        return np.vstack([multivariate_normal(
            self.centers[cl], self.cov[cl], 1)[0] for cl in c])


class class_dist:

    def __init__(self, n_classes=2, dim=2, n_clusters_per_class=4):
        self.dists = [multi_gauss(n_clusters=n_clusters_per_class, dim=dim)
                      for _ in range(n_classes)]
        self.n_classes = n_classes
        self.dim = dim

    def sample(self, label):
        return self.dists[label].sample()

    def create_network(self, n_samples, labels, seed=4423):
        np.random.seed(seed)
        pos = [self.sample(label) for label in labels]
        pos = dict((i, pos[i]) for i in range(len(pos)))
        g = random_geometric_graph(n_samples, radius=0.2,
                                   dim=self.dim, pos=pos)
        return g, labels

    def create_random_label_network(self, n_samples, seed=243):
        np.random.seed(seed)
        return self.create_network(n_samples, randint(0, self.n_classes, n_samples))

    def show_sample_network(self, n_samples):
        labels = randint(0, self.n_classes, n_samples)
        pos = [self.sample(label) for label in labels]
        pos = dict((i, pos[i]) for i in range(len(pos)))
        g = random_geometric_graph(n_samples, radius=0.2, dim=self.dim, pos=pos)
        nx.draw_networkx(g, pos=pos, node_color=labels, with_labels=False, node_size=100)
        plt.draw()
        thr.start_new(plt.show, ())


def create_class_dist_sample(n_samples=400, n_classes=2, n_rel_true=2, n_rel_false=2, n_feat_true=2, n_feat_false=2, dim=2, n_clusters_per_class=4, seed=34):
    s = seed
    np.random.seed(s)
    rel = []
    labels = randint(0, n_classes, n_samples)
    for _ in range(n_rel_true):
        r, _ = class_dist(n_classes=n_classes, dim=dim, n_clusters_per_class=n_clusters_per_class).create_network(n_samples, labels, seed=s)
        rel.append(r)
        s = s + 1
    for _ in range(n_rel_false):
        r, _ = class_dist(n_classes=n_classes, dim=dim, n_clusters_per_class=n_clusters_per_class).create_random_label_network(n_samples, seed=s)
        rel.append(r)
        s = s + 1
    features = np.zeros(shape=(n_samples, 0))
    if n_feat_true > 0:
        cd = class_dist(n_classes=n_classes, dim=dim, n_clusters_per_class=n_clusters_per_class)
        features = np.vstack([cd.sample(label) for label in labels])
    if n_feat_false > 0:
        cd = class_dist(n_classes=n_classes, dim=dim, n_clusters_per_class=n_clusters_per_class)
        f2 = np.vstack([cd.sample(label) for label in randint(0, n_classes, n_samples)])
        features = np.hstack([features, f2])
    return rel, DataFrame(features), DataFrame(labels, columns=['labels'])

def show_graph(net, labels):
    nx.draw_spring(net, node_color=labels, with_labels=False, node_size=100)
    plt.draw()
    plt.show()

def show(net, labels):
    show_graph(nx.to_networkx_graph(net), labels)

def show_points(points, labels):
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'rx')
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'b.')
    thr.start_new(plt.show, ())
