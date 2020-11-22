from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from model import GCNModelVAE,GCNModelVAECD,GCNModelAE
from utils import load_data, preprocess_graph, get_roc_score, sparse_to_tuple,sparse_mx_to_torch_sparse_tensor,cluster_acc,clustering_evaluation
from preprocessing import mask_test_feas,mask_test_edges, load_AN
from tqdm import tqdm
from tensorboardX import SummaryWriter
from evaluation import clustering_latent_space
from collections import Counter

import warnings
warnings.simplefilter("ignore")

def training(args):

    print("Using {} dataset".format(args.dataset_str))
    adj_init, features, Y= load_AN(args.dataset_str)
    print("imported graph edge number:{}".format(adj_init.sum()))
    assert adj_init.diagonal().sum()==0,"adj diagonal sum should be 0"
    n_nodes, n_features= features.shape

    args.nClusters=len(set(Y))
    # args.nClusters=1
    print("cluster number:{}".format(args.nClusters))
    assert(adj_init.shape[0]==n_nodes)

    print("node size:{}, feature size:{}".format(n_nodes,n_features))


    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_init)

    # print("graph edge number after mask:{}".format(adj_init.sum()))



    # save result to files
    link_predic_result_file = "result/AGAE_{}.res".format(args.dataset_str)
    embedding_node_mean_result_file = "result/AGAE_{}_n_mu.emb".format(args.dataset_str)
    embedding_attr_mean_result_file = "result/AGAE_{}_a_mu.emb".format(args.dataset_str)
    embedding_node_var_result_file = "result/AGAE_{}_n_sig.emb".format(args.dataset_str)
    embedding_attr_var_result_file = "result/AGAE_{}_a_sig.emb".format(args.dataset_str)

    # Some preprocessing, get the support matrix, D^{-1/2}\hat{A}D^{-1/2}
    adj_norm = preprocess_graph(adj_init)
    print("graph edge number after normalize adjacent matrix:{}".format(adj_init.sum()))

    pos_weight_u = torch.tensor(float(adj_init.shape[0] * adj_init.shape[0] - adj_init.sum()) / adj_init.sum()) #??
    norm_u = adj_init.shape[0] * adj_init.shape[0] / float((adj_init.shape[0] * adj_init.shape[0] - adj_init.sum()) * 2) #??
    # norm_u = 1
	pos_weight_a = torch.tensor(float(features[2][0] * features[2][1] - len(features[1])) / len(features[1]))
	norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)

    features_training = sparse_mx_to_torch_sparse_tensor(features)

    #clustering pretraining for GMM paramter initialization
    writer=SummaryWriter('./logs')

    adj_label = torch.FloatTensor(adj_init.toarray()+sp.eye(adj_init.shape[0])) # add the identity matrix to the adj as label
	features_label = features

    mean_h=[]
    mean_c=[]
    mean_v=[]
    mean_ari=[]
    mean_ami=[]
    mean_nmi=[]
    mean_purity=[]

    for r in range(args.num_run):

        model = None
        if args.model == 'gcn_ae':
            model = GCNModelAE(n_features,n_nodes, args.hidden1, args.hidden2, args.dropout,args)
        elif args.model == 'gcn_vae':
            model = GCNModelVAE(n_features,n_nodes, args.hidden1, args.hidden2, args.dropout,args)
        elif args.model == 'gcn_vaecd':
            model = GCNModelVAECD(n_features,n_nodes, args.hidden1, args.hidden2, args.dropout,args)
		elif args.model =='gcn_vaece': #gcn with vae for co-embedding of feature and graph
            model = GCNModelVAECE(n_features,n_nodes, args.hidden1, args.hidden2, args.dropout,args)

            # using GMM to pretrain the  clustering parameters
            # model.pre_train(features_training,adj_norm,Y,pre_epoch=20)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)


        hidden_emb_u = None
        hidden_emb_a = None

        cost_val = []
        acc_val = []
        val_roc_score = []
        # lr_s=StepLR(optimizer,step_size=10,gamma=0.95)

        for epoch in range(args.epochs):
            t = time.time()
            model.train()

            if args.model =='gcn_vaecd':
                recovered_u, mu_u, logvar_u = model(features_training, adj_norm)
                loss = model.loss(features_training,adj_norm,labels = adj_label, n_nodes = n_nodes, n_features = n_features,norm = norm_u, pos_weight = pos_weight_u)

            elif args.model == 'gcn_ae':
                recovered_u, mu_u,logvar_u = model(features_training, adj_norm)
                loss = model.loss(recovered_u,labels = adj_label, n_nodes = n_nodes, n_features = n_features,norm = norm_u, pos_weight = pos_weight_u)
            elif args.model == 'gcn_vae':
                recovered_u, mu_u, logvar_u = model(features_training, adj_norm)
                loss = model.loss(features_training,adj_norm,labels = adj_label, n_nodes = n_nodes, n_features = n_features,norm = norm_u, pos_weight = pos_weight_u)
			elif args.model =='gcn_vaece': #gcn with vae for co-embedding of feature and graph
				loss = model.loss(features_training,adj_norm,labels = (adj_label, features_label), n_nodes = n_nodes, n_features = n_features,norm = (norm_u, norm_a), pos_weight = (pos_weight_u, pos_weight_a))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_s.step()

            # model.check_parameters()

            correct_prediction_u = ((torch.sigmoid(recovered_u)>=0.5)==adj_label.type(torch.LongTensor))
            # correct_prediction_a = ((torch.sigmoid(recovered_a)>=0.5).type(torch.LongTensor)==features_label.type(torch.LongTensor)).type(torch.FloatTensor)

            accuracy = torch.mean(correct_prediction_u*1.0)

            # hidden_emb_u = mu_u.data.numpy()
            # hidden_emb_a = mu_a.data.numpy()
            # roc_curr, ap_curr = get_roc_score(np.dot(hidden_emb_u,hidden_emb_u.T), adj, val_edges, val_edges_false)
            # roc_curr_a, ap_curr_a = get_roc_score(np.dot(hidden_emb_u,hidden_emb_a.T), features_orig, val_feas, val_feas_false)

            # val_roc_score.append(roc_curr)

            #clustering#############
            pre=[]
            tru=[]


            tru=Y
            model.eval()

            # if args.model == 'vgaecd':
                # pre=model.predict(mu_u,logvar_u)

                # print("True label:{}".format(tru))
                # print(Counter(tru))
                # print("Predicted label:{}".format(pre))
                # print(Counter(pre))

                # # mc_
                # print("cluster means")
                # print(model.mu_c.data)

                # print("cluster prior")
                # print(model.pi_.data)
            # else:
                # pre=clustering_latent_space(mu_u.detach().numpy(),tru)

            writer.add_scalar('loss',loss.item(),epoch)
            # writer.add_scalar('acc',cluster_acc(pre,tru)[0]*100,epoch)
            # writer.add_scalar('lr',lr_s.get_last_lr()[0],epoch)

            # print('Loss={:.4f},Clustering_ACC={:.4f}%,LR={:.4f}'.format(loss.item(),cluster_acc(pre,tru)[0]*100,lr_s.get_last_lr()[0]))
            # H, C, V, ari, ami, nmi, purity  = clustering_evaluation(tru,pre)
            # print('H:{} C:{} V:{} ari:{} ami:{} nmi:{} purity:{}'.format(H, C, V, ari, ami, nmi, purity))

            #######################


            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(loss.item()),
                  # "log_lik=", "{:.5f}".format(cost.item()),
                  # "KL_u=", "{:.5f}".format(KLD_u.item()),
                  # "KL_a=", "{:.5f}".format(KLD_a.item()),
                  # "yita_loss=", "{:.5f}".format(yita_loss.item()),
                  "link_pred_train_acc=", "{:.5f}".format(accuracy.item()),
                  # "val_edge_roc=", "{:.5f}".format(val_roc_score[-1]),
                  # "val_edge_ap=", "{:.5f}".format(ap_curr),
                  # "val_attr_roc=", "{:.5f}".format(roc_curr_a),
                  # "val_attr_ap=", "{:.5f}".format(ap_curr_a),
                  "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")

        recovered_u, mu_u, logvar_u = model(features_training, adj_norm)
        pre=clustering_latent_space(mu_u.detach().numpy(),tru)
        H, C, V, ari, ami, nmi, purity  = clustering_evaluation(tru,pre)
        mean_h.append(round(H,4))
        mean_c.append(round(C,4))
        mean_v.append(round(V,4))
        mean_ari.append(round(ari,4))
        mean_ami.append(round(ami,4))
        mean_nmi.append(round(nmi,4))
        mean_purity.append(round(purity,4))

        # np.save(embedding_node_mean_result_file, mu_u.data.numpy())
        # np.save(embedding_attr_mean_result_file, mu_a.data.numpy())
        # np.save(embedding_node_var_result_file, logvar_u.data.numpy())
        # np.save(embedding_attr_var_result_file, logvar_a.data.numpy())

        # roc_score, ap_score = get_roc_score(np.dot(hidden_emb_u,hidden_emb_u.T), adj, test_edges, test_edges_false)
        # roc_score_a, ap_score_a = get_roc_score(np.dot(hidden_emb_u,hidden_emb_a.T), features_orig, test_feas, test_feas_false)

        # print('Test edge ROC score: ' + str(roc_score))
        # print('Test edge AP score: ' + str(ap_score))
        # print('Test attr ROC score: ' + str(roc_score_a))
        # print('Test attr AP score: ' + str(ap_score_a))
    ###### Report Final Results ######
    print('Homogeneity:{}\t mean:{}\t std:{}\n'.format(mean_h,round(np.mean(mean_h),4),round(np.std(mean_h),4)))
    print('Completeness:{}\t mean:{}\t std:{}\n'.format(mean_c,round(np.mean(mean_c),4),round(np.std(mean_c),4)))
    print('V_measure_score:{}\t mean:{}\t std:{}\n'.format(mean_v,round(np.mean(mean_v),4),round(np.std(mean_v),4)))
    print('adjusted Rand Score:{}\t mean:{}\t std:{}\n'.format(mean_ari,round(np.mean(mean_ari),4),round(np.std(mean_ari),4)))
    print('adjusted Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_ami,round(np.mean(mean_ami),4),round(np.std(mean_ami),4)))
    print('Normalized Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_nmi,round(np.mean(mean_nmi),4),round(np.std(mean_nmi),4)))
    print('Purity:{}\t mean:{}\t std:{}\n'.format(mean_purity,round(np.mean(mean_purity),4),round(np.std(mean_purity),4)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_ae', help="models used for clustering: gcn_ae,gcn_vae,gcn_vaecd")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nClusters',type=int,default=7)
    parser.add_argument('--num_run',type=int,default=1,help='Number of running times')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    training(args)
