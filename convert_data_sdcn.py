import numpy as np
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
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
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    return adj, features, labels, idx_train, idx_val, idx_test

def load_AN(dataset):
    edge_file = open("data/{}.edge".format(dataset), 'r')
    attri_file = open("data/{}.node".format(dataset), 'r')
    label_file = open("data/{}.label".format(dataset),'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    labels_raw = label_file.readlines()
    edge_file.close()
    attri_file.close()
    label_file.close()

    node_num = int(edges[0].split()[1].strip())
    edge_num = int(edges[1].split()[1].strip())
    attribute_number = int(attributes[1].split()[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    edge_num_no_selfloop= 0
    for line in edges:
        node1 = int(line.split()[0].strip())
        node2 = int(line.split()[1].strip())
        if node1==node2:
            continue
        adj_row.append(node1)
        adj_col.append(node2)
        edge_num_no_selfloop+=1
    adj = sp.csc_matrix((np.ones(edge_num_no_selfloop), (adj_row, adj_col)), shape=(node_num, node_num))

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split()[0].strip())
        attribute1 = int(line.split()[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))

    labels=[]
    for label in labels_raw:
        label = int(label.strip())
        labels.append(label)
    return adj, attribute, np.array(labels)

dataset_name = sys.argv[1]
if dataset_name in ['cora','citeseer']:
    adj, attribute, y, idx_train, idx_val, idx_test = load_data(dataset_name)

    y = np.argmax(y,1) # labels is in one-hot format
    attribute = attribute.toarray()

    wp = open('data/{}.txt'.format(dataset_name),'w')
    for i in range(attribute.shape[0]):
        for j in range(attribute.shape[1]):
            wp.write("{} ".format(int(attribute[i][j])))
        wp.write('\n')
    wp.close()

    wp = open('data/{}_label.txt'.format(dataset_name),'w')

    for i in range(y.shape[0]):
        wp.write('{}\n'.format(y[i]))
    wp.close()

    wp = open('data/{}_graph.txt'.format(dataset_name),'w')
    adj = adj.toarray()
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j]==1:
                wp.write("{} {}\n".format(i,j))
    wp.close()

else:
    adj,attribute, y = load_AN(dataset_name)

    attribute = attribute.toarray()

    wp = open('data/{}.txt'.format(dataset_name),'w')
    for i in range(attribute.shape[0]):
        for j in range(attribute.shape[1]):
            wp.write("{} ".format(int(attribute[i][j])))
        wp.write('\n')




