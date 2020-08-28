from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, preprocess_graph, get_roc_score, sparse_to_tuple,sparse_mx_to_torch_sparse_tensor,cluster_acc,clustering_evaluation
from preprocessing import mask_test_feas,mask_test_edges, load_AN
from tqdm import tqdm
from tensorboardX import SummaryWriter

import warnings
warnings.simplefilter("ignore")
from collections import Counter

def training(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, Y= load_AN(args.dataset_str)
    # adj, features = load_data(args.dataset_str)
    n_nodes, n_features= features.shape

    args.nClusters=len(set(Y))
    print("cluster number:{}".format(args.nClusters))
    assert(adj.shape[0]==n_nodes)

    print("node size:{}, feature size:{}".format(n_nodes,n_features))

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)

    # adj = adj_train
    adj = adj_orig

    features_orig = features
    features_label = torch.FloatTensor(features.toarray())
    features = sp.lil_matrix(features)

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # save result to files
    link_predic_result_file = "result/AGAE_{}.res".format(args.dataset_str)
    embedding_node_mean_result_file = "result/AGAE_{}_n_mu.emb".format(args.dataset_str)
    embedding_attr_mean_result_file = "result/AGAE_{}_a_mu.emb".format(args.dataset_str)
    embedding_node_var_result_file = "result/AGAE_{}_n_sig.emb".format(args.dataset_str)
    embedding_attr_var_result_file = "result/AGAE_{}_a_sig.emb".format(args.dataset_str)

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj + sp.eye(adj.shape[0]) # in preprocess_graph the adj added with the diagonal matrix
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight_u = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm_u = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # pos_weight_a = torch.tensor(float(features[2][0] * features[2][1] - len(features[1])) / len(features[1]))
    # norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)

    features_training = sparse_mx_to_torch_sparse_tensor(features_orig)

    model = GCNModelVAE(n_features,n_nodes, args.hidden1, args.hidden2, args.dropout,args)

    #check parameter order
    model.check_parameters()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_s=StepLR(optimizer,step_size=10,gamma=0.95)


    hidden_emb_u = None
    hidden_emb_a = None

    cost_val = []
    acc_val = []
    val_roc_score = []

    #clustering pretraining for GMM paramter initialization
    writer=SummaryWriter('./logs')
    # model.pre_train(features_training,adj_norm,Y,pre_epoch=50)

    # epoch_bar=tqdm(range(args.epochs))
    # print(model.training)
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        recovered_u, mu_u, logvar_u = model(features_training, adj_norm)

        cost_u,KLD_u,yita_loss = model.ELBO_loss(features_training,adj_norm,labels = adj_label, n_nodes = n_nodes, n_features = n_features,norm = norm_u, pos_weight = pos_weight_u)
        # cost_u, cost_a, KLD_u, KLD_a = loss_function(preds = (recovered_u, recovered_a), labels = (adj_label, features_label),
                             # m = (mu_u, mu_a), logvar = (logvar_u, logvar_a),  n_nodes = n_nodes, n_features = n_features,
        # loss= L_rec_u+L_rec_a+KLD_u_c+KLD_a+yita_loss                    # norm = (norm_u, norm_a), pos_weight = (pos_weight_u, pos_weight_a))
        loss = cost_u + 10*KLD_u + yita_loss
        loss.backward()
        optimizer.step()

        lr_s.step()

        # model.check_parameters()

        correct_prediction_u = ((torch.sigmoid(recovered_u)>=0.5).type(torch.LongTensor)==adj_label.type(torch.LongTensor)).type(torch.FloatTensor)
        # correct_prediction_a = ((torch.sigmoid(recovered_a)>=0.5).type(torch.LongTensor)==features_label.type(torch.LongTensor)).type(torch.FloatTensor)

        accuracy = torch.mean(correct_prediction_u)

        hidden_emb_u = mu_u.data.numpy()
        # hidden_emb_a = mu_a.data.numpy()
        roc_curr, ap_curr = get_roc_score(np.dot(hidden_emb_u,hidden_emb_u.T), adj_orig, val_edges, val_edges_false)
        # roc_curr_a, ap_curr_a = get_roc_score(np.dot(hidden_emb_u,hidden_emb_a.T), features_orig, val_feas, val_feas_false)

        val_roc_score.append(roc_curr)

        #clustering#############
        pre=[]
        tru=[]


        tru=Y
        model.eval()
        pre=model.predict(mu_u,logvar_u)
        print("True label:{}".format(tru))
        print(Counter(tru))
        print("Predicted label:{}".format(pre))
        print(Counter(pre))

        # mc_
        print("cluster means")
        print(model.mu_c.data)

        print("cluster prior")
        print(model.pi_.data)

        writer.add_scalar('loss',loss.item(),epoch)
        writer.add_scalar('acc',cluster_acc(pre,tru)[0]*100,epoch)
        writer.add_scalar('lr',lr_s.get_last_lr()[0],epoch)

        print('Loss={:.4f},Clustering_ACC={:.4f}%,LR={:.4f}'.format(loss.item(),cluster_acc(pre,tru)[0]*100,lr_s.get_last_lr()[0]))
        print('H C V ari ami nmi purity {}'.format(clustering_evaluation(tru,pre)))

        #######################


        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(loss.item()),
              "log_lik=", "{:.5f}".format(cost_u.item()),
              "KL_u=", "{:.5f}".format(KLD_u.item()),
              # "KL_a=", "{:.5f}".format(KLD_a.item()),
              "yita_loss=", "{:.5f}".format(yita_loss.item()),
              "link_pred_train_acc=", "{:.5f}".format(accuracy.item()),
              "val_edge_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_edge_ap=", "{:.5f}".format(ap_curr),
              # "val_attr_roc=", "{:.5f}".format(roc_curr_a),
              # "val_attr_ap=", "{:.5f}".format(ap_curr_a),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")
    # np.save(embedding_node_mean_result_file, mu_u.data.numpy())
    # np.save(embedding_attr_mean_result_file, mu_a.data.numpy())
    # np.save(embedding_node_var_result_file, logvar_u.data.numpy())
    # np.save(embedding_attr_var_result_file, logvar_a.data.numpy())

    roc_score, ap_score = get_roc_score(np.dot(hidden_emb_u,hidden_emb_u.T), adj_orig, test_edges, test_edges_false)
    # roc_score_a, ap_score_a = get_roc_score(np.dot(hidden_emb_u,hidden_emb_a.T), features_orig, test_feas, test_feas_false)

    print('Test edge ROC score: ' + str(roc_score))
    print('Test edge AP score: ' + str(ap_score))
    # print('Test attr ROC score: ' + str(roc_score_a))
    # print('Test attr AP score: ' + str(ap_score_a))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=8, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nClusters',type=int,default=10)
    parser.add_argument('--hid_dim',type=int,default=8)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    training(args)
