import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
import pickle

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def load_multilayer_dataset(dataset_name, train_percent=10, seed=123):
    if dataset_name == 'vickers':
        return load_vickers_data(train_percent=train_percent, seed=seed)
    elif dataset_name == 'leskovec-ng':
        return load_leskovec_data(train_percent=train_percent, seed=seed)
    elif dataset_name == 'congress-votes':
        return load_congress_votes_data(train_percent=train_percent, seed=seed)
    elif dataset_name == 'cora':
        return load_cora_multilayer(train_percent=train_percent, seed=seed)
    elif dataset_name == 'ABIDE':
        return load_ABIDE_data(train_percent=train_percent, seed=seed)
    elif dataset_name == 'reinnovation':
        return load_reinnovation_data(train_percent=train_percent, seed=seed)
    elif dataset_name == 'balance_scale':
        return load_balance_scale_data(train_percent=train_percent, seed=seed)
    elif dataset_name == 'mammography':
        return load_mammographic_masses_data(train_percent=train_percent, seed=seed)
    else:
        raise ValueError('Dataset name not defined.')


'''
    Create bidirectional edges (or undirected) for all datasets
    Move this to the pre-processing step in the notebook
    # Get edge list.
    # G = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())  # Use DiGraph to get edges in both directions.
    # edgelist = np.array(G.edges())
    
    # Add self loop.
    # N = adj.shape[0]
    # for i in range(0, N):
    #     adj[i, i] = 1

'''


def format_edgelists(edgelists, num_nodes):

    # Include edges in both directions.
    for layer, edgelist in enumerate(edgelists):
        new_edgelist = edgelist.tolist()
        for edge in edgelist:
            # Check for reverse edge.
            inverse_direction = [edge[1], edge[0]]
            if inverse_direction not in new_edgelist:
                new_edgelist.append(inverse_direction)

            # Check for self-edge.
            self_edge = [edge[0], edge[0]]
            if self_edge not in new_edgelist:
                new_edgelist.append(self_edge)

        # Deal with number of nodes < num_nodes in specific layers.
        for node in range(num_nodes):
            isolated_node_edge = [node, node]
            if isolated_node_edge not in new_edgelist:
                new_edgelist.append(isolated_node_edge)

        new_edgelist.sort()
        edgelists[layer] = np.array(new_edgelist)

    return edgelists


def load_vickers_data(train_percent=10, seed=123,feature_dim=3):
    file_name = 'data/Multilayer Graph Datasets/Vickers/Vickers.Chan.7thGraders.multiplex.edges'
    data = np.loadtxt(fname=file_name, usecols=(0, 1, 2), dtype=int)
    data = data - 1  # Make layer and node index start from 0.

    edge_list0 = data[data[:, 0] == 0, 1:]
    edge_list1 = data[data[:, 0] == 1, 1:]
    edge_list2 = data[data[:, 0] == 2, 1:]

    N = len(np.unique(edge_list0[:, 0]))

    features = np.random.randn(N, feature_dim)
    y = np.vstack((np.zeros((12, 1), dtype=np.int8), np.ones((17, 1), dtype=np.int8)))
    y = y.reshape((y.shape[0]))

    y_true = np.eye(2)[np.squeeze(y)]

    train_nodes =   round(N * (train_percent/100))# number of train_nodes per class
    test_nodes = N - train_nodes
    train_mask = np.concatenate((np.ones(4), np.zeros(20), np.ones(5))).astype(dtype=bool)
    val_mask = np.concatenate((np.zeros(4), np.ones(20), np.zeros(5))).astype(dtype=bool)
    # TODO: validation and test nodes are the same.
    test_mask = np.concatenate((np.zeros(4), np.ones(20), np.zeros(5))).astype(dtype=bool)

    edge_lists = [edge_list0, edge_list1, edge_list2]

    #train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    y_train = np.zeros((N, 2))
    y_val = np.zeros((N, 2))
    y_test = np.zeros((N, 2))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    edge_lists = format_edgelists(edge_lists, N)

    return edge_lists, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def load_dataset(multiplex_edge_file=None,
                 multiplex_lables_file=None,
                 multiplex_features_file=None,
                 train_percent=10,
                 seed=123):

    assert multiplex_edge_file is not None
    assert multiplex_lables_file is not None

    multiplex_edges = np.loadtxt(fname=multiplex_edge_file, usecols=(0, 1, 2), dtype=int)
    multiplex_labels = np.loadtxt(fname=multiplex_lables_file, dtype=int)

    if multiplex_features_file is not None:
        multiplex_features = np.loadtxt(fname=multiplex_features_file)
    else:
        multiplex_features = None

    P = len(set(multiplex_edges[:,0]))  # num of multi-layers
    N = max(multiplex_edges[:,1]) + 1   # num of nodes - node ids start from 0
    C = len(set(multiplex_labels))

    edge_lists = []
    for i in range(P):
        edge_list = multiplex_edges[multiplex_edges[:, 0] == i, 1:]
        edge_lists.append(edge_list)

    y_true = np.eye(C)[np.squeeze(multiplex_labels)]

    train_nodes = int(round(N * (train_percent / 100)))  # number of train_nodes per class
    test_nodes = N - train_nodes

    idx_train = np.concatenate((np.ones(train_nodes), np.zeros(test_nodes))).astype(dtype=np.int8)
    idx_val = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)
    idx_test = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)

    train_mask = idx_train.astype(dtype=bool)
    val_mask = idx_val.astype(dtype=bool)
    test_mask = idx_test.astype(dtype=bool)

    train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    y_train = np.zeros((N, C))
    y_val = np.zeros((N, C))
    y_test = np.zeros((N, C))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    return edge_lists, multiplex_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, multiplex_labels


def load_leskovec_data(train_percent=10, seed=123):
    file_name = 'data/Multilayer Graph Datasets/Leskovec-Ng Dataset/Leskovec-Ng.multilayer.edges'
    data = np.loadtxt(fname=file_name).astype(np.int32)

    edge_list0 = data[data[:, 0] == 0, 1:]
    edge_list1 = data[data[:, 0] == 1, 1:]
    edge_list2 = data[data[:, 0] == 2, 1:]
    edge_list3 = data[data[:, 0] == 3, 1:]

    # TODO: move the preprocessing to the notebook
    '''if dataset_name == 'leskovec-ng':
        temp_array = np.column_stack((np.array(range(0, N)), np.array(range(0, N))))
        for i, el in enumerate(edgelists_L):
            edgelists_L[i] = np.append(el, temp_array, axis=0)
    '''
    N = 191

    # print(edge_list0)
    features = np.random.randn(N, 1)

    y = np.loadtxt(fname='data/Multilayer Graph Datasets/Leskovec-Ng Dataset/Leskovec-Ng.multilayer.labels').astype(np.int32)
    y = y - 1
    y_true = np.eye(2)[np.squeeze(y)]

    train_nodes = round(N * (train_percent/100))  # number of train_nodes per class
    test_nodes = N - train_nodes

    # idx_train = np.concatenate((np.ones(train_nodes), np.zeros(test_nodes), np.ones(train_nodes))).astype(dtype=np.int8)
    # idx_val = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes), np.zeros(train_nodes))).astype(dtype=np.int8)
    # idx_test = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes), np.zeros(train_nodes))).astype(dtype=np.int8)

    idx_train = np.concatenate((np.ones(train_nodes), np.zeros(test_nodes))).astype(dtype=np.int8)
    idx_val = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)
    idx_test = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)

    train_mask = idx_train.astype(dtype=bool)
    val_mask = idx_val.astype(dtype=bool)
    test_mask = idx_test.astype(dtype=bool)

    train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    edge_lists = [edge_list0, edge_list1, edge_list2, edge_list3]

    y_train = np.zeros((N, 2))
    y_val = np.zeros((N, 2))
    y_test = np.zeros((N, 2))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    return edge_lists, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def load_congress_votes_data(train_percent=10, seed=111):
    file_name = 'data/Multilayer Graph Datasets/109th-Congress/congress-votes.multilayer.edges'
    data = np.loadtxt(fname=file_name).astype(np.int32)

    y = np.loadtxt(fname='data/Multilayer Graph Datasets/109th-Congress/congress-votes.multilayer.labels').astype(np.int32)
    y_true = np.eye(2)[np.squeeze(y)]

    num_layers = np.max(data[:, 0]) + 1
    N = y_true.shape[0]

    edge_lists = []
    for i in range(num_layers):
        edge_lists.append(data[data[:, 0] == i, 1:])

    features = np.random.randn(N, 1)

    train_nodes = round(N * (train_percent/100))
    test_nodes = N - train_nodes

    idx_train = np.concatenate((np.ones(train_nodes), np.zeros(test_nodes))).astype(dtype=np.int8)
    idx_val = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)
    idx_test = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)

    train_mask = idx_train.astype(dtype=bool)
    val_mask = idx_val.astype(dtype=bool)
    test_mask = idx_test.astype(dtype=bool)

    train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    y_train = np.zeros((N, 2))
    y_val = np.zeros((N, 2))
    y_test = np.zeros((N, 2))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    return edge_lists[0:4], features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def load_balance_scale_data(train_percent=10, seed=111):
    file_name = 'data/Multilayer Graph Datasets/Balance_Scale/balance_scale.multilayer.edges'
    data = np.loadtxt(fname=file_name).astype(np.int32)

    y = np.loadtxt(fname='data/Multilayer Graph Datasets/Balance_Scale/balance_scale.multilayer.labels').astype(
        np.int32)
    y_true = np.eye(3)[np.squeeze(y)]

    num_layers = np.max(data[:, 0]) + 1
    N = y_true.shape[0]

    edge_lists = []
    for i in range(num_layers):
        edge_lists.append(data[data[:, 0] == i, 1:])

    features = np.random.randn(N, 1)

    train_nodes = round(N * (train_percent/100))
    print('train nodes ', train_nodes)
    test_nodes = N - train_nodes

    idx_train = np.concatenate((np.ones(train_nodes), np.zeros(test_nodes))).astype(dtype=np.int8)
    idx_val = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)
    idx_test = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)

    train_mask = idx_train.astype(dtype=bool)
    val_mask = idx_val.astype(dtype=bool)
    test_mask = idx_test.astype(dtype=bool)

    train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    y_train = np.zeros((N, 3))
    y_val = np.zeros((N, 3))
    y_test = np.zeros((N, 3))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    return edge_lists, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def load_mammographic_masses_data(train_percent=10, seed = 123):
    file_name = 'data/Multilayer Graph Datasets/Mammographic_masses/mammographic_masses.multilayer.edges'
    data = np.loadtxt(fname=file_name).astype(np.int32)

    y = np.loadtxt(fname='data/Multilayer Graph Datasets/Mammographic_masses/mammographic_masses.multilayer.labels').astype(
        np.int32)
    y_true = np.eye(2)[np.squeeze(y)]

    num_layers = np.max(data[:, 0]) + 1
    N = y_true.shape[0]

    edge_lists = []
    for i in range(num_layers):
        edge_lists.append(data[data[:, 0] == i, 1:])

    features = np.random.randn(N, 1)

    train_nodes = round(N * (train_percent/100))
    test_nodes = N - train_nodes

    idx_train = np.concatenate((np.ones(train_nodes), np.zeros(test_nodes))).astype(dtype=np.int8)
    idx_val = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)
    idx_test = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)

    train_mask = idx_train.astype(dtype=bool)
    val_mask = idx_val.astype(dtype=bool)
    test_mask = idx_test.astype(dtype=bool)

    train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    y_train = np.zeros((N, 2))
    y_val = np.zeros((N, 2))
    y_test = np.zeros((N, 2))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    return edge_lists, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def load_cora_multilayer(train_percent=10, seed=123):
    adj1, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
    for i in range(0, adj1.shape[0]):
        adj1[i, i] = 1
    # Get edge list.
    G1 = nx.from_scipy_sparse_matrix(adj1, create_using=nx.DiGraph())  # Use DiGraph to get edges in both directions.
    edge_list_1 = np.array(G1.edges())

    features = preprocess_features(features)
    features = features.todense()
    adj2 = kneighbors_graph(X=features, n_neighbors=200, mode='connectivity', include_self=True)

    # Get edge list.
    G2 = nx.from_scipy_sparse_matrix(adj2, create_using=nx.DiGraph())  # Use DiGraph to get edges in both directions.
    edge_list_2 = np.array(G2.edges())

    N = y_train.shape[0]
    features = np.random.randn(N, 1)

    edge_lists = [edge_list_1, edge_list_2, ]
    y = np.argmax(y_train, axis=1) + np.argmax(y_val, axis=1) + np.argmax(y_test, axis=1)
    print('y_train' + str(np.argmax(y_train, axis=1)))
    print(y.shape)
    print(y)

    return edge_lists, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def load_ABIDE_data(train_percent=10, seed=123):
    file_name = 'data/Multilayer Graph Datasets/ABIDE/abide.multilayer.edges'
    data = np.loadtxt(fname=file_name).astype(np.int32)

    y = np.loadtxt(fname='data/Multilayer Graph Datasets/ABIDE/abide.multilayer.labels').astype(np.int32)
    y_true = np.eye(2)[np.squeeze(y)]

    num_layers = np.max(data[:, 0]) + 1
    N = y_true.shape[0]

    edge_lists = []
    for i in range(num_layers):
        edge_lists.append(data[data[:, 0] == i, 1:])

    features = np.loadtxt(fname='data/Multilayer Graph Datasets/ABIDE/abide.multilayer.features').astype(np.float32)

    path = 'data/Multilayer Graph Datasets/ABIDE/tr_ids.pkl'
    with open(path, 'rb') as file_obj:
        train_ids = pickle.load(file_obj, encoding='latin1')

    idx_train = np.zeros(features.shape[0])
    idx_train[train_ids[5]] = 1

    train_mask = idx_train.astype(dtype=bool)
    val_mask = np.logical_not(idx_train)
    test_mask = np.logical_not(idx_train)

    #train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    y_train = np.zeros((N, 2))
    y_val = np.zeros((N, 2))
    y_test = np.zeros((N, 2))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    return edge_lists, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def load_reinnovation_data(train_percent=10, seed=123):
    file_name = 'data/Multilayer Graph Datasets/Reinnovation/reinnovation.multilayer.edges'
    data = np.loadtxt(fname=file_name).astype(np.int32)

    y = np.loadtxt(fname='data/Multilayer Graph Datasets/Reinnovation/reinnovation.multilayer.labels').astype(np.int32)
    y_true = np.eye(3)[np.squeeze(y)]

    num_layers = np.max(data[:, 0]) + 1
    N = y_true.shape[0]

    edge_lists = []
    for i in range(num_layers):
        edge_lists.append(data[data[:, 0] == i, 1:])

    features = np.random.randn(N, 1)

    print(num_layers)

    train_nodes = round(N * (train_percent/100))
    test_nodes = N - train_nodes

    idx_train = np.concatenate((np.ones(train_nodes), np.zeros(test_nodes))).astype(dtype=np.int8)
    idx_val = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)
    idx_test = np.concatenate((np.zeros(train_nodes), np.ones(test_nodes))).astype(dtype=np.int8)

    train_mask = idx_train.astype(dtype=bool)
    val_mask = idx_val.astype(dtype=bool)
    test_mask = idx_test.astype(dtype=bool)

    train_mask, val_mask, test_mask = random_permutation(N, train_mask, val_mask, test_mask, seed=seed)

    y_train = np.zeros((N, 3))
    y_val = np.zeros((N, 3))
    y_test = np.zeros((N, 3))

    y_train[train_mask, :] = y_true[train_mask, :]
    y_val[val_mask, :] = y_true[val_mask, :]
    y_test[test_mask, :] = y_true[test_mask, :]

    return edge_lists[0:4], features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y


def degree_from_edgelist(edgelist):
    num_nodes = np.max(np.unique(edgelist)) + 1
    d = [edgelist[edgelist[:, 0] == node, 1].shape[0] for node in range(num_nodes)]

    return np.array(d).reshape((num_nodes, 1))


def make_unique_node_ID(edgelists, num_nodes):
    assert num_nodes is not None
    edge_List_unique_ID = []
    for p, edgelist in enumerate(edgelists):
        edge_List_unique_ID.append(edgelist + (p * num_nodes))

    return edge_List_unique_ID


def create_supragraph_edgelist(edgelists_L, N):
    edgelists_L_unique_ID = make_unique_node_ID(edgelists_L, N)
    num_layers = len(edgelists_L)

    # Create a supra-edgelist
    supra_edges_List = []  # creating pillar edges
    for i in range(num_layers):
        for j in range(num_layers):
            if i == j:
                continue

            pillar_edgelist = np.column_stack((np.array(range(i * N, (i + 1) * N)),
                                               np.array(range(j * N, (j + 1) * N))))
            supra_edges_List.append(pillar_edgelist)

    # edgelist consisting only of pillar edges
    supra_graph_edgelist = np.concatenate(supra_edges_List, axis=0)
    # edgelist consisting of all the original in-layer edges
    in_layer_edgelist = np.concatenate(edgelists_L_unique_ID, axis=0)

    # Creating one big edgelist
    all_edgelist = np.concatenate([supra_graph_edgelist, in_layer_edgelist], axis=0)
    return all_edgelist


def plot_train_validation_loss_accuracy(cost_train=None,
                                        acc_train=None,
                                        cost_val=None,
                                        acc_val=None,
                                        title_string=None,
                                        dataset_name=None):

    # plot settings
    font_size = 13
    color = 'blue'
    line_width = 2

    savepath = './Plots/' + dataset_name + '_results.pdf'
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    plt.plot(cost_train, color=color, linewidth=line_width)
    plt.title('Training Loss value', fontsize=font_size)
    plt.xlabel('Epoch')

    plt.subplot(222)
    plt.plot(acc_train, color=color, linewidth=line_width)
    plt.title('Training Accuracy', fontsize=font_size)
    plt.xlabel('Epoch')

    plt.subplot(223)
    plt.plot(cost_val, color=color, linewidth=line_width)
    plt.title('Validation Loss value', fontsize=font_size)
    plt.xlabel('Epoch')

    plt.subplot(224)
    plt.plot(acc_val, color=color, linewidth=line_width)
    plt.title('Validation Accuracy', fontsize=font_size)
    plt.xlabel('Epoch')

    plt.suptitle(title_string,fontsize=font_size, fontweight='bold')
    #plt.show()
    plt.savefig(savepath)
    return


def plot_tsne_embeddings(labels, mask, features, dataset_name):

    # plot parameter settings
    marker_style = ','
    marker_size = 6
    annotate_size = 6
    # t-SNE plot
    savepath =  './Plots/' + dataset_name + '_embeddings.pdf'
    tsne = TSNE(n_components=2)
    test_labels = np.argmax(labels[mask], axis=1)  # Consider test embeddings.
    test_features_L = [feat[mask] for feat in features]

    # Initial learned embeddings.
    epoch = 1
    test_features_epoch_1 = test_features_L[epoch-1]
    #print(type(test_features_epoch_1))
    #print(np.array(test_features_epoch_1))
    feat = tsne.fit_transform(features[0][mask])
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(121)
    ax1.scatter(feat[:, 0], feat[:, 1], c=test_labels, marker=marker_style, s=marker_size)
    ax1.set_title('Epoch number {0}'.format(epoch))

    # Final learned embeddings.
    epoch = len(features)
    feat = tsne.fit_transform(test_features_L[epoch - 1])
    ax2 = fig2.add_subplot(122)
    ax2.scatter(feat[:, 0], feat[:, 1], c=test_labels, marker=marker_style, s=marker_size)
    ax2.set_title('Epoch number {0}'.format(epoch))

    # plt.show()
    plt.savefig(savepath)
    return

def construct_Baseline_feed_dict(edgelists_L, features, y_labels, labels_mask, place_holders, attention_dropout=1,
                                 input_dropout=1):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({place_holders['input_features']: features})
    feed_dict.update({place_holders['edgelists_L'][i]: edgelists_L[i] for i in range(len(edgelists_L))})
    feed_dict.update({place_holders['labels']: y_labels})
    feed_dict.update({place_holders['labels_mask']: labels_mask})
    feed_dict.update({place_holders['attention_dropout_keep']: attention_dropout})
    feed_dict.update({place_holders['input_dropout_keep']: input_dropout})

    return feed_dict


def construct_Supra_feed_dict(supra_edgelist, supra_features, y_labels, labels_mask, place_holders,
                              attention_dropout=1, input_dropout=1):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({place_holders['supra_input_features']: supra_features})
    feed_dict.update({place_holders['supra_edgelist']: supra_edgelist})
    feed_dict.update({place_holders['labels']: y_labels})
    feed_dict.update({place_holders['labels_mask']: labels_mask})
    feed_dict.update({place_holders['attention_dropout_keep']: attention_dropout})
    feed_dict.update({place_holders['input_dropout_keep']: input_dropout})

    return feed_dict


def construct_single_layer_feed_dict(edgelist, features, y_labels, labels_mask, place_holders,
                                     attention_dropout=1, input_dropout=1):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({place_holders['edgelist']: edgelist})
    feed_dict.update({place_holders['input_features']: features})
    feed_dict.update({place_holders['labels']: y_labels})
    feed_dict.update({place_holders['labels_mask']: labels_mask})
    feed_dict.update({place_holders['attention_dropout_keep']: attention_dropout})
    feed_dict.update({place_holders['input_dropout_keep']: input_dropout})

    return feed_dict


def flatten_lists(lists):
    return [item for list in lists for item in list]


def random_permutation(num_nodes=None, train_mask=None, val_mask=None, test_mask=None, seed=123):
    """ Performs random permutation on the data effectively
    changing the train, val and test nodes"""

    np.random.seed(seed)

    permutation = list(np.random.permutation(num_nodes))
    train_mask = train_mask[permutation]
    val_mask = val_mask[permutation]
    test_mask = test_mask[permutation]

    return train_mask, val_mask, test_mask


