import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
import os
from tqdm import tqdm

from layers import GraphConvolution, GraphConvolutionSparse, Linear, InnerDecoder
from utils import cluster_acc


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout,args):
        super(GCNModelVAE, self).__init__()

        self.args = args
        self.gc1 = GraphConvolutionSparse(input_feat_dim, hidden_dim1, dropout, act=torch.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dc = InnerDecoder(dropout, act=lambda x: x)

        #for embedding attributes/features
        # self.linear_a1= Linear(n_nodes, hidden_dim1, act = torch.tanh,sparse_inputs=True) # the input dim is the number of nodes
        # self.linear_a2= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)
        # self.linear_a3= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)


        self.pi_=nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1)/args.nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.randn(args.nClusters,args.hid_dim),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.randn(args.nClusters,args.hid_dim),requires_grad=True)

    def encoder(self, x, adj):
        hidden1 = self.gc1(x, adj)
        # hidden_a1 = self.linear_a1(x.t()) # transpose the input feature matrix

        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def decoder(self,mu,logvar):

        z_u = self.reparameterize(mu, logvar)
        # z_a = self.reparameterize(mu_a,logvar_a)

        return self.dc(z_u)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):

        mu, logvar = self.encoder(x, adj)
        z_u = self.reparameterize(mu, logvar)
        # z_a = self.reparameterize(mu_a,logvar_a)
        return self.dc(z_u),mu, logvar


    def ELBO_loss(self,x,adj,labels, n_nodes, n_features, norm, pos_weight,L=1):

        det=1e-10
        labels_sub_u = labels
        norm_u = norm
        pos_weight_u= pos_weight

        L_rec_u=0

        mu, logvar = self.encoder(x, adj)
        # z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):

            # z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu
            pred_adj = self.decoder(mu,logvar)
            # L_rec+=F.binary_cross_entropy(x_pro,x)

            # cost_u = norm * F.binary_cross_entropy_with_logits(pred_adj, labels_sub_u,pos_weight = pos_weight)
            cost_u = norm * F.binary_cross_entropy_with_logits(pred_adj, labels_sub_u,pos_weight = pos_weight)
            # cost_a = norm_a * F.binary_cross_entropy_with_logits(pred_x, labels_sub_a, pos_weight = pos_weight_a)
            # cost_a =torch.Tensor(1).fill_(0)

            L_rec_u += cost_u
            # L_rec_a += cost_a

        L_rec_u/=L
        # L_rec_a/=L

        # z_a = self.reparameterize(mu_a,logvar_a)
        # KLD_a = (0.5 / n_features) * torch.mean(torch.sum(-1 - 2 * logvar_a + mu_a.pow(2) + logvar_a.exp().pow(2), 1))
        # KLD_a =torch.Tensor(1).fill_(0)

        # Loss=L_rec*x.size(1)


        self.pi_.data = (self.pi_/self.pi_.sum()).data
        # log_sigma2_c=self.log_sigma2_c
        # mu_c=self.mu_c

        # z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        z = self.reparameterize(mu,logvar)

        yita_c=torch.exp(torch.log(self.pi_.unsqueeze(0))+self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
        yita_c = F.softmax(yita_c)

        # yita_c=yita_c/(yita_c.sum(1).view(-1,1)) #shape: batch_size*Clusters

        KLD_u_c=(0.5 / n_nodes)*torch.mean(torch.sum(yita_c*torch.sum(self.log_sigma2_c.unsqueeze(0)+\
        torch.exp(2*logvar.unsqueeze(1)-self.log_sigma2_c.unsqueeze(0))+\
        (mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.log_sigma2_c.unsqueeze(0)),2),1))

        # KLD_u_c-= (0.5/n_nodes)*torch.mean(torch.sum(1+2*logvar,1))
        yita_loss = (1 / self.args.nClusters) * torch.mean(torch.sum(yita_c*torch.log(yita_c/self.pi_.unsqueeze(0)),1)) - (0.5 / self.args.hid_dim)*torch.mean(torch.sum(1+2*logvar,1))

        return L_rec_u,KLD_u_c,yita_loss

    def pre_train(self,x,adj,Y,pre_epoch=50):
        '''
        This function is used to initialize  cluster paramters: pi_, mu_c, log_sigma2_c.
        -------------
        paramters:
        x: is the feature matrix of graph G.
        adj: is the adjacent matrix of graph G.
        Y: is the class label for each node in graph G.
        '''

        if  not os.path.exists('./pretrain_model_{}.pk'.format(self.args.dataset_str)):

            Loss=nn.MSELoss()
            opti=Adam(self.parameters()) #all paramters in model

            print('Pretraining......')
            # epoch_bar=tqdm(range(pre_epoch))
            # for _ in epoch_bar:
            for _ in range(pre_epoch):

                self.train()
                L=0
                mu, logvar  = self.encoder(x,adj)
                pred_adj = self.decoder(mu,logvar)

                loss=  Loss(pred_adj,adj.to_dense())

                L+=loss.detach().cpu().numpy()

                opti.zero_grad()
                loss.backward()
                opti.step()

                # epoch_bar.write('L2={:.4f}'.format(L))
                print('L2={:.4f}'.format(L))

            self.gc2.load_state_dict(self.gc3.state_dict())
            # self.linear_a2.load_state_dict(self.linear_a3.state_dict())


            with torch.no_grad():
                    mu, logvar  = self.encoder(x,adj)
                    assert F.mse_loss(mu, logvar) == 0
                    # assert F.mse_loss(mu_a, logvar_a) == 0
                    Z = mu.data.numpy()


            gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')

            pre = gmm.fit_predict(Z)
            print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            self.pi_.data = torch.from_numpy(gmm.weights_).float()
            self.mu_c.data = torch.from_numpy(gmm.means_).float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).float())

            torch.save(self.state_dict(), './pretrain_model_{}.pk'.format(self.args.dataset_str))
        else:
            self.load_state_dict(torch.load('./pretrain_model_{}.pk'.format(self.args.dataset_str)))

    def predict(self,mu, logvar):
        # z_mu, z_sigma2_log, z_ma,z_a_sigma2_log = self.encoder(x,adj)
        # mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
        # z = torch.randn_like(mu) * torch.exp(logvar) + mu
        z  = self.reparameterize(mu,logvar)
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()

        return np.argmax(yita,axis=1)


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.args.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)


    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def check_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data,param.data.shape)

