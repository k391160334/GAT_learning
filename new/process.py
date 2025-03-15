import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = True
    return mask


def load_data(dataset_str):
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open(f"../data/ind.{dataset_str}.{names[i]}", "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"../data/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(features.todense())

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # labels = torch.LongTensor(np.argmax(labels, axis=1))

    idx_test = torch.LongTensor(test_idx_range)
    idx_train = torch.LongTensor(range(len(y)))
    idx_val = torch.LongTensor(range(len(y), len(y) + 500))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = torch.LongTensor(labels[idx_train])
    y_val = torch.LongTensor(labels[idx_val])
    y_test = torch.LongTensor(labels[idx_test])

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv = np.where(np.isinf(r_inv), 0.0, r_inv)
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    # 이미 numpy.ndarray라면 .todense() 대신 바로 torch로 변환
    if sp.issparse(features):
        features = features.todense()

    return torch.FloatTensor(features)
