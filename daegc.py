from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from model import DAEGCE
from utils import preprocess_graph, get_roc_score, sparse_to_tuple,sparse_mx_to_torch_sparse_tensor,cluster_acc,clustering_evaluation, find_motif,drop_feature, drop_edge,choose_cluster_votes,plot_tsne,save_results,entropy_metric
from preprocessing import mask_test_feas,mask_test_edges, load_AN, check_symmetric,load_data
from tqdm import tqdm
from tensorboardX import SummaryWriter
from evaluation import clustering_latent_space
from collections import Counter
import itertools
import random
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hungrian import label_mapping

import warnings
warnings.simplefilter("ignore")

def training(args):

    print("Using {} dataset".format(args.dataset))
    if args.dataset in ['cora','pubmed','citeseer']:
        adj_init, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
        Y = np.argmax(labels,1) # labels is in one-hot format
    else:
        adj_init, features, Y= load_AN(args.dataset)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_init = adj_init - sp.dia_matrix((adj_init.diagonal()[np.newaxis, :], [0]), shape=adj_init.shape)
    adj_init.eliminate_zeros()

    assert adj_init.diagonal().sum()==0,"adj diagonal sum:{}, should be 0".format(adj_init.diagonal().sum())
    n_nodes, n_features= features.shape
    # assert check_symmetric(adj_init).sum()==n_nodes*n_nodes,"adj should be symmetric"
    print("imported graph edge number (without selfloop):{}".format((adj_init-adj_init.diagonal()).sum()/2))


    args.nClusters=len(set(Y))
    # args.nClusters=1
    print("cluster number:{}".format(args.nClusters))
    assert(adj_init.shape[0]==n_nodes)

    print("node size:{}, feature size:{}".format(n_nodes,n_features))


    # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_init)
    # fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)

    features_orig = features
    features_label = torch.FloatTensor(features.toarray())
    features = sp.lil_matrix(features)

    features = sparse_to_tuple(features.tocoo())

    features_nonzero = features[1].shape[0]

    print("graph edge number after mask:{}".format(adj_init.sum()/2))



    # save result to files
    link_predic_result_file = "result/AGAE_{}.res".format(args.dataset)
    embedding_node_mean_result_file = "result/AGAE_{}_n_mu.emb".format(args.dataset)
    embedding_attr_mean_result_file = "result/AGAE_{}_a_mu.emb".format(args.dataset)
    embedding_node_var_result_file = "result/AGAE_{}_n_sig.emb".format(args.dataset)
    embedding_attr_var_result_file = "result/AGAE_{}_a_sig.emb".format(args.dataset)

    # Some preprocessing, get the support matrix, D^{-1/2}\hat{A}D^{-1/2}
    adj_norm = preprocess_graph(adj_init)
    print("graph edge number after normalize adjacent matrix:{}".format(adj_init.sum()/2))

    pos_weight_u = torch.tensor(float(adj_init.shape[0] * adj_init.shape[0] - adj_init.sum()) / adj_init.sum()) #??
    norm_u = adj_init.shape[0] * adj_init.shape[0] / float((adj_init.shape[0] * adj_init.shape[0] - adj_init.sum()) * 2) #??
    pos_weight_a = torch.tensor(float(features[2][0] * features[2][1] - len(features[1])) / len(features[1]))
    norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)

    features_training = sparse_mx_to_torch_sparse_tensor(features_orig)

    # clustering pretraining for GMM paramter initialization
    # writer=SummaryWriter('./logs')

    adj_label = torch.FloatTensor(adj_init.toarray()+sp.eye(adj_init.shape[0])) # add the identity matrix to the adj as label

    mean_h=[]
    mean_c=[]
    mean_v=[]
    mean_ari=[]
    mean_ami=[]
    mean_nmi=[]
    mean_purity=[]
    mean_accuracy=[]
    mean_f1=[]
    mean_precision=[]
    mean_recall = []
    mean_entropy = []


    if args.cuda:
        features_training = features_training.to_dense().cuda() # it needs higher memory if to_dense
        adj_norm = adj_norm.to_dense().cuda()
        pos_weight_u = pos_weight_u.cuda()
        pos_weight_a = pos_weight_a.cuda()
        adj_label = adj_label.cuda()
        features_label = features_label.cuda()

    features_training, adj_norm = Variable(features_training), Variable(adj_norm)
    pos_weight_u = Variable(pos_weight_u)
    pos_weight_a = Variable(pos_weight_a)

    for r in range(args.num_run):

        # random.seed(args.seed)
        # np.random.seed(args.seed)
        # torch.manual_seed(args.seed)

        model = DAEGCE(n_features,n_nodes, args.hidden1, args.hidden2, args.dropout,args)

        if args.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)


        hidden_emb_u = None
        hidden_emb_a = None

        cost_val = []
        acc_val = []
        val_roc_score = []
        lr_s=StepLR(optimizer,step_size=30,gamma=1) # it seems that fix leanring rate is better

        loss_list=None
        pretrain_flag = False
        for epoch in range(args.epochs):
            t = time.time()
            model.train()

            loss_list,[z]= model.loss(features_training,adj_norm, adj_label, n_nodes, n_features, norm_u, pos_weight_u, epoch)

            pre, gamma = model.predict_soft_assignment(z)

            H, C, V, ari, ami, nmi, purity, f1_score,precision,recall = clustering_evaluation(Y,pre)
            print("purity, NMI, f1_score:",purity,nmi,f1_score)

            if epoch <200:
                loss =loss_list[0] # when epoch < T1=200, only update reconstruction loss
            else:
                if pretrain_flag == False:
                    pretrain_flag = True
                    print('pre_train',pretrain_flag)
                    kmeans= KMeans(n_clusters=args.nClusters)
                    pre = kmeans.fit_predict(z.cpu().detach().numpy())
                    H, C, V, ari, ami, nmi, purity, f1_score,precision,recall = clustering_evaluation(Y,pre)
                    print("kmeans purity, NMI:",purity,nmi)
                    plot_tsne(args.dataset,args.model,epoch,z.cpu(),model.mu_c.cpu(),Y,pre)
                    model.init_clustering_params_kmeans(kmeans)

                loss =loss_list[0]+loss_list[1]


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recovered_u = model(features_training, adj_norm)

            lr_s.step()


            correct_prediction_u = ((torch.sigmoid(recovered_u.to('cpu'))>=0.5)==adj_label.type(torch.LongTensor))

            accuracy = torch.mean(correct_prediction_u*1.0)


            #clustering#############
            pre=[]
            tru=[]
            gamma = None


            tru=Y

            print("Epoch:", '%04d' % (epoch + 1),
                "LR={:.4f}".format(lr_s.get_last_lr()[0]),
                  "train_loss_total=", "{:.5f}".format(loss.item()),
                  "train_loss_parts=", "{}".format([round(l.item(),4) for l in loss_list]),
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

        pre,gamma_c = model.predict_soft_assignment(z)

        print("label mapping using Hungarian algorithm ")
        pre = label_mapping(tru,pre)

        with open("./DAEGCE/{}_{}_prediction.log".format(args.model,args.dataset),'w') as wp:
            for label in pre:
                wp.write("{}\n".format(label))

        print('gamma_c:',gamma_c)
        print('gamma_c argmax:',np.argmax(gamma_c,1))
        print('gamma_c argmax counter:',Counter(np.argmax(gamma_c,1).tolist()))

        H, C, V, ari, ami, nmi, purity, f1_score,precision,recall = clustering_evaluation(Y,pre)
        entropy = entropy_metric(tru,pre)
        acc = cluster_acc(pre,tru)[0]*100
        mean_h.append(round(H,4))
        mean_c.append(round(C,4))
        mean_v.append(round(V,4))
        mean_ari.append(round(ari,4))
        mean_ami.append(round(ami,4))
        mean_nmi.append(round(nmi,4))
        mean_purity.append(round(purity,4))
        mean_accuracy.append(round(acc,4))
        mean_f1.append(round(f1_score,4))
        mean_precision.append(round(precision,4))
        mean_recall.append(round(recall,4))
        mean_entropy.append(round(entropy,4))

        plot_tsne(args.dataset,args.model,epoch,z.cpu(),model.mu_c.cpu(),tru,pre)


    metrics_list=[mean_h,mean_c,mean_v,mean_ari,mean_ami,mean_nmi,mean_purity,mean_accuracy,mean_f1,mean_precision,mean_recall,mean_entropy]
    save_results(args.dataset,args.model,args.epochs,metrics_list)
    ###### Report Final Results ######
    print('Homogeneity:{}\t mean:{}\t std:{}\n'.format(mean_h,round(np.mean(mean_h),4),round(np.std(mean_h),4)))
    print('Completeness:{}\t mean:{}\t std:{}\n'.format(mean_c,round(np.mean(mean_c),4),round(np.std(mean_c),4)))
    print('V_measure_score:{}\t mean:{}\t std:{}\n'.format(mean_v,round(np.mean(mean_v),4),round(np.std(mean_v),4)))
    print('adjusted Rand Score:{}\t mean:{}\t std:{}\n'.format(mean_ari,round(np.mean(mean_ari),4),round(np.std(mean_ari),4)))
    print('adjusted Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_ami,round(np.mean(mean_ami),4),round(np.std(mean_ami),4)))
    print('Normalized Mutual Information:{}\t mean:{}\t std:{}\n'.format(mean_nmi,round(np.mean(mean_nmi),4),round(np.std(mean_nmi),4)))
    print('Purity:{}\t mean:{}\t std:{}\n'.format(mean_purity,round(np.mean(mean_purity),4),round(np.std(mean_purity),4)))
    print('Accuracy:{}\t mean:{}\t std:{}\n'.format(mean_accuracy,round(np.mean(mean_accuracy),4),round(np.std(mean_accuracy),4)))
    print('F1-score:{}\t mean:{}\t std:{}\n'.format(mean_f1,round(np.mean(mean_f1),4),round(np.std(mean_f1),4)))
    print('precision_score:{}\t mean:{}\t std:{}\n'.format(mean_precision,round(np.mean(mean_precision),4),round(np.std(mean_precision),4)))
    print('recall_score:{}\t mean:{}\t std:{}\n'.format(mean_recall,round(np.mean(mean_recall),4),round(np.std(mean_recall),4)))
    print('entropy:{}\t mean:{}\t std:{}\n'.format(mean_entropy,round(np.mean(mean_entropy),4),round(np.std(mean_entropy),4)))
    print("True label distribution:{}".format(tru))
    print(Counter(tru))
    print("Predicted label distribution:{}".format(pre))
    print(Counter(pre))

def parse_args():
    parser = argparse.ArgumentParser(description="Node clustering")
    parser.add_argument('--model', type=str, default='daegce', help="models used for clustering: gcn_ae,gcn_vae,gcn_vaecd,gcn_vaece")
    parser.add_argument('--seed', type=int, default=20, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=256, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial aearning rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--nClusters',type=int,default=7)
    parser.add_argument('--num_run',type=int,default=1,help='Number of running times')
    parser.add_argument('--cuda', action='store_true', default=False, help='Disables CUDA training.')
    args, unknown = parser.parse_known_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cuda:
        torch.cuda.set_device(0)
        # torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    training(args)
