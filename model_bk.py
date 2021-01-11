import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from tqdm import tqdm

from layers import GraphConvolution, GraphConvolutionSparse, Linear, InnerDecoder, InnerProductDecoder
from utils import cluster_acc

from utils_smiles import *
from estimators import estimate_mutual_information
from collections import Counter

class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout,args):
        super(GCNModelAE, self).__init__()

        self.args = args
        self.gc1 = GraphConvolutionSparse(input_feat_dim, hidden_dim1, dropout, act=torch.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        # self.dc = InnerDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        z = self.gc1(x,adj)
        z = self.gc2(z,adj)
        return self.dc(z),z,None


    def loss(self,pred_adj,labels, n_nodes, n_features, norm, pos_weight,L=1):

        cost = norm * F.binary_cross_entropy_with_logits(pred_adj, labels,pos_weight = pos_weight)
        return cost,

    def check_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data,param.data.shape)

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout,args):
        super(GCNModelVAE, self).__init__()

        self.args = args
        self.gc1 = GraphConvolutionSparse(input_feat_dim, hidden_dim1, dropout, act=torch.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        # self.dc = InnerDecoder(dropout, act=lambda x: x)


    def encoder(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def decoder(self,mu,logvar):

        z_u = self.reparameterize(mu, logvar)

        return self.dc(z_u)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

        # if self.training:
            # std = torch.exp(logvar)
            # eps = torch.randn_like(std)
            # return eps.mul(std).add_(mu)
        # else:
            # return mu

    def forward(self, x, adj):

        mu, logvar = self.encoder(x, adj)
        z_u = self.reparameterize(mu, logvar)
        # z_a = self.reparameterize(mu_a,logvar_a)
        return self.dc(z_u),mu, logvar


    def loss(self,x,adj,labels, n_nodes, n_features, norm, pos_weight,L=1):

        det=1e-10
        norm_u = norm
        pos_weight_u= pos_weight

        L_rec_u=0

        mu, logvar = self.encoder(x, adj)
        # z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):

            pred_adj = self.decoder(mu,logvar)

            cost_u = norm * F.binary_cross_entropy_with_logits(pred_adj, labels ,pos_weight = pos_weight)

            L_rec_u += cost_u

        L_rec_u/=L

        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2),1))
        return L_rec_u, KLD


    def check_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data,param.data.shape)


class GCNModelVAECD(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout,args):
        super(GCNModelVAECD, self).__init__()

        self.args = args
        self.gc1 = GraphConvolutionSparse(input_feat_dim, hidden_dim1, dropout, act=torch.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        # self.dc = InnerDecoder(dropout, act=lambda x: x)

        #for embedding attributes/features
        # self.linear_a1= Linear(n_nodes, hidden_dim1, act = torch.tanh,sparse_inputs=True) # the input dim is the number of nodes
        # self.linear_a2= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)
        # self.linear_a3= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)


        self.pi_=nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1)/args.nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.randn(args.nClusters,hidden_dim2),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.randn(args.nClusters,hidden_dim2),requires_grad=True)

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


    def loss(self,x,adj,labels, n_nodes, n_features, norm, pos_weight,L=1):

        det=1e-10
        norm_u = norm
        pos_weight_u= pos_weight

        L_rec_u=0

        mu, logvar = self.encoder(x, adj)
        hidden_dim2 = mu.shape[1]

        # z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):

            # z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu
            pred_adj = self.decoder(mu,logvar)
            # L_rec+=F.binary_cross_entropy(x_pro,x)

            # cost_u = norm * F.binary_cross_entropy_with_logits(pred_adj, labels_sub_u,pos_weight = pos_weight)
            cost_u = norm * F.binary_cross_entropy_with_logits(pred_adj, labels ,pos_weight = pos_weight)
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

        gamma_c=torch.exp(torch.log(self.pi_.unsqueeze(0))+self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
        gamma_c = F.softmax(gamma_c) # is softmax a good way?

        gamma_c=gamma_c/(gamma_c.sum(1).view(-1,1)) #shape: batch_size*Clusters
        self.pi_.data = gamma_c.mean(0).data # prior need to be re-normalized? In GMM, prior is based on gamma_c:https://brilliant.org/wiki/gaussian-mixture-model/

        # KLD_u_c=(0.5 / n_nodes)*torch.mean(torch.sum(gamma_c*torch.sum(self.log_sigma2_c.unsqueeze(0)+\
            # torch.exp(2*logvar.unsqueeze(1)-self.log_sigma2_c.unsqueeze(0))+\
            # (mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.log_sigma2_c.unsqueeze(0)),2),1))

        # KLD_u_c-= (0.5/n_nodes)*torch.mean(torch.sum(1+2*logvar,1))
        # gamma_loss = (1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c/self.pi_.unsqueeze(0)),1)) - (0.5 / hidden_dim2)*torch.mean(torch.sum(1+2*logvar,1))

        KLD_u_c=-(0.5/n_nodes)*torch.mean(torch.sum(gamma_c*torch.sum(-1+self.log_sigma2_c.unsqueeze(0)-2*logvar.unsqueeze(1)+
            torch.exp(2*logvar.unsqueeze(1)-self.log_sigma2_c.unsqueeze(0))+
            (mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.log_sigma2_c.unsqueeze(0)),2),1))

        gamma_loss = -(1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c/self.pi_.unsqueeze(0)),1))

        return L_rec_u,-KLD_u_c,-gamma_loss

    def pre_train(self,x,adj,Y,pre_epoch=10):
        '''
        This function is used to initialize  cluster paramters: pi_, mu_c, log_sigma2_c.
        -------------
        paramters:
        x: is the feature matrix of graph G.
        adj: is the adjacent matrix of graph G.
        Y: is the class label for each node in graph G.
        '''

        if  not os.path.exists('./pretrain_model_{}.pk'.format(self.args.dataset)):

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

            torch.save(self.state_dict(), './pretrain_model_{}.pk'.format(self.args.dataset))
        else:
            self.load_state_dict(torch.load('./pretrain_model_{}.pk'.format(self.args.dataset)))

    def predict(self,mu, logvar):
        # z_mu, z_sigma2_log, z_ma,z_a_sigma2_log = self.encoder(x,adj)
        # mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
        # z = torch.randn_like(mu) * torch.exp(logvar) + mu
        z  = self.reparameterize(mu,logvar)
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        gamma_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        gamma=gamma_c.detach().cpu().numpy()

        return np.argmax(gamma,axis=1),gamma


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

class GCNModelVAECE(nn.Module):
    def __init__(self, input_feat_dim, n_nodes, hidden_dim1, hidden_dim2, dropout,args):
        super(GCNModelVAECE, self).__init__()


        self.args = args
        self.gc1 = GraphConvolutionSparse(input_feat_dim, hidden_dim1, dropout, act=torch.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.dc = InnerDecoder(dropout, act=lambda x: x)

        #for embedding attributes/features
        self.linear_a1= Linear(n_nodes, hidden_dim1, act = torch.tanh,sparse_inputs=True) # the input dim is the number of nodes
        self.linear_a2= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)
        self.linear_a3= Linear(hidden_dim1, hidden_dim2, act = lambda x:x)

        #modularity layer
        self.modulairty_layer = Linear(hidden_dim2,args.nClusters,act=torch.relu)
        # cluster choosing
        self.cluster_choose= Linear(hidden_dim2,args.nClusters,act=torch.sigmoid)


        self.pi_=nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1)/args.nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(args.nClusters,hidden_dim2).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(args.nClusters,hidden_dim2).fill_(1),requires_grad=True)

        # torch.nn.init.xavier_normal_(self.mu_c)
        # torch.nn.init.xavier_normal_(self.log_sigma2_c)

        # calculate mi

        # critic_params = {'dim_x': x.shape[1],'dim_y':y.shape[1],'layers': 2,'embed_dim': 32,'hidden_dim': 64,'activation': 'relu',}
        # self.critic_structure = ConcatCritic(hidden_dim2,n_nodes,256,3,'relu',rho=None,)
        # self.critic_feature = ConcatCritic(hidden_dim2,input_feat_dim,256,3,'relu',rho=None,)

    def encoder(self, x, adj):
        hidden1 = self.gc1(x, adj)
        hidden_a1 = self.linear_a1(x.t()) # transpose the input feature matrix
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), self.linear_a2(hidden_a1),self.linear_a3(hidden_a1)

    def decoder(self,mu,mu_a,logvar,logvar_a):

        z_u = self.reparameterize(mu, logvar)
        z_a = self.reparameterize(mu_a,logvar_a)
        return self.dc((z_u,z_a))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):

        mu, logvar, mu_a, logvar_a = self.encoder(x, adj)
        z_u = self.reparameterize(mu, logvar)
        z_a = self.reparameterize(mu_a,logvar_a)
        return self.dc((z_u,z_a)),mu, logvar, mu_a, logvar_a

    def modularity_loss(self, z,adj):

        adj = adj.to_dense()
        H = self.modulairty_layer(z)
        assert H.shape[0]==z.shape[0]

        n = torch.tensor(1.0*z.shape[0])

        H_norm = n.sqrt()*H.sqrt()/(H.sqrt().sum())
        print("H_norm shape",H_norm.shape)
        print("H_norm ",H_norm)
        m = (adj-torch.eye(adj.shape[0])).sum()/2
        D = (adj-torch.eye(adj.shape[0])).sum(1) # the degree of nodes, adj includes self loop
        B = (adj-torch.eye(adj.shape[0]))-torch.matmul(D.view(-1,1),D.view(1,-1))/(2*m) # modularity matrix
        mod_loss=torch.trace(torch.matmul(torch.matmul(H_norm.t(),B),H_norm)/(4*m))
        print("mod_loss",mod_loss)

        return mod_loss

    def dist(self,x):
        # x = x/torch.norm(x,2,dim=1).view(-1,1)
        assert len(x.size()) == 2
        norm = (x ** 2).sum(1).view(-1, 1)
        dn = (norm + norm.view(1, -1)) - 2.0 * (x @ x.t())
        return torch.sum(torch.relu(dn).sqrt())

    def mi_loss(self,z,x,a):
        # critic_params = {'dim_x': x.shape[1],'dim_y':y.shape[1],'layers': 2,'embed_dim': 32,'hidden_dim': 64,'activation': 'relu',}
        # critic = ConcatCritic(rho=None,**critic_params)
        indice = torch.randperm(len(z))[0:50]
        # mi_x = estimate_mutual_information('dv',z[indice],x[indice],self.critic_structure)
        mi_a = estimate_mutual_information('js',z[indice],a[indice],self.critic_feature)
        return mi_a

    def change_cluster_grad_false(self):
        for name, param in self.named_parameters():
            if name in ['pi_','mu_c','log_sigma2_c']:
                param.requires_grad=False

    def change_cluster_grad_true(self):
        for name, param in self.named_parameters():
            if name in ['pi_','mu_c','log_sigma2_c']:
                param.requires_grad=True


    def change_nn_grad_false(self):
        for name, param in self.named_parameters():
            if name not in ['pi_','mu_c','log_sigma2_c']:
                param.requires_grad=False

    def change_nn_grad_true(self):
        for name, param in self.named_parameters():
            if name not in ['pi_','mu_c','log_sigma2_c']:
                param.requires_grad=True

    def loss(self,x,adj,labels, n_nodes, n_features, norm, pos_weight,L=1):

        det=1e-10
        labels_sub_u, labels_sub_a = labels
        norm_u, norm_a = norm
        pos_weight_u, pos_weight_a = pos_weight

        L_rec_u=0
        L_rec_a=0

        mi=0

        mu, logvar, mu_a, logvar_a = self.encoder(x, adj)

        # mutual information loss

        # z_mu, z_sigma2_log = self.encoder(x)
        # mi_a = self.mi_loss(mu,adj.to_dense(),x.to_dense())
        for l in range(L):

            # z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu
            pred_adj, pred_x = self.decoder(mu,mu_a,logvar,logvar_a)
            # L_rec+=F.binary_cross_entropy(x_pro,x)

            cost_u = norm_u * F.binary_cross_entropy_with_logits(pred_adj, labels_sub_u, pos_weight = pos_weight_u)
            cost_a = norm_a * F.binary_cross_entropy_with_logits(pred_x, labels_sub_a, pos_weight = pos_weight_a)
            # cost_a =torch.Tensor(1).fill_(0)

            L_rec_u += cost_u
            L_rec_a += cost_a


        L_rec_u/=L
        L_rec_a/=L

        # z_a = self.reparameterize(mu_a,logvar_a)
        # KLD_a = (0.5 / n_features) * torch.mean(torch.sum(-1 - 2 * logvar_a + mu_a.pow(2) + logvar_a.exp().pow(2), 1))
        KLD_a = -(0.5 / n_features) * torch.mean(torch.sum(-1 - 2 * logvar_a + mu_a.pow(2) + logvar_a.exp().pow(2), 1))
        # KLD_a =torch.Tensor(1).fill_(0)

        # Loss=L_rec*x.size(1)


        # log_sigma2_c=self.log_sigma2_c
        # mu_c=self.mu_c

        # z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        z = self.reparameterize(mu,logvar)

        # mod_loss=self.modularity_loss(z,adj)
        # gamma_c=torch.exp(torch.log(self.pi_.unsqueeze(0))+self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
        # gamma_c=torch.exp(self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
        gamma_c  = self.cluster_choose(z)
        # print('gamma_c:',gamma_c)

        gamma_c=gamma_c/(gamma_c.sum(1).view(-1,1))#batch_size*Clusters
        gamma_c=F.softmax(gamma_c)
        # print('gamma_c normalized:',gamma_c)
        # print('gamma_c argmax:',torch.argmax(gamma_c,1))
        print('gamma_c counter:',Counter(torch.argmax(gamma_c,1).tolist()))


        # self.pi_.data = (self.pi_/self.pi_.sum()).data # prior need to be re-normalized? In GMM, prior is based on gamma_c:https://brilliant.org/wiki/gaussian-mixture-model/
        # self.pi_.data = gamma_c.mean(0).data # prior need to be re-normalized? In GMM, prior is based on gamma_c:https://brilliant.org/wiki/gaussian-mixture-model/

        KLD_u_c=-(0.5/n_nodes)*torch.mean(torch.sum(gamma_c*torch.sum(-1+self.log_sigma2_c.unsqueeze(0)-2*logvar.unsqueeze(1)+torch.exp(2*logvar.unsqueeze(1)-self.log_sigma2_c.unsqueeze(0))+(mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.log_sigma2_c.unsqueeze(0)),2),1))

        # KLD_u_c_test=-(0.5/n_nodes)*F.mse_loss(mu.unsqueeze(1),self.mu_c.unsqueeze(0),reduction='none')
        # print('kld_u_c_test:',KLD_u_c_test.sum(2))


        # KLD_u_c=-(0.5/n_nodes)*F.mse_loss(mu.unsqueeze(1),self.mu_c.unsqueeze(0))

        # KLD_u_c=(0.5 / n_nodes)*torch.mean(torch.sum(gamma_c*torch.sum(self.log_sigma2_c.unsqueeze(0)+\
            # torch.exp(2*logvar.unsqueeze(1)-self.log_sigma2_c.unsqueeze(0))+\
            # (mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.log_sigma2_c.unsqueeze(0)),2),1))

        mutual_dist = (1/(self.args.nClusters**2))*self.dist(self.mu_c)

        gamma_loss=-(1/self.args.nClusters)*torch.mean(torch.sum(gamma_c*torch.log(gamma_c),1))
        # gamma_loss = (1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c),1)) - (0.5 / self.args.hid_dim)*torch.mean(torch.sum(1+2*logvar,1))
        # gamma_loss = -(1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c/self.pi_.unsqueeze(0)),1))
        # gamma_loss = (1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c/self.pi_.unsqueeze(0)),1)) - (0.5 / self.args.hid_dim)*torch.mean(torch.sum(1+2*logvar,1))


        # return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a
        return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a , -gamma_loss, -0.05*mutual_dist
        # return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a , -gamma_loss,-mi_a
        # return L_rec_u + L_rec_a + KLD_u_c + KLD_a + gamma_loss


    def pre_train(self,x,adj,Y,pre_epoch=20):
        '''
        This function is used to initialize  cluster paramters: pi_, mu_c, log_sigma2_c.
        -------------
        paramters:
        x: is the feature matrix of graph G.
        adj: is the adjacent matrix of graph G.
        Y: is the class label for each node in graph G.
        '''

        if not os.path.exists('./pretrain_model_{}.pk'.format(self.args.dataset)):

            Loss=nn.MSELoss()
            opti=Adam(self.parameters()) #all paramters in model

            print('Pretraining......')
            # epoch_bar=tqdm(range(pre_epoch))
            # for _ in epoch_bar:
            for _ in range(pre_epoch):

                self.train()
                L=0
                mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
                pred_adj, pred_x = self.decoder(mu,mu_a,logvar,logvar_a)

                loss=  Loss(pred_x,x.to_dense()) + Loss(pred_adj,adj.to_dense())

                L+=loss.detach().cpu().numpy()

                opti.zero_grad()
                loss.backward()
                opti.step()

                # epoch_bar.write('L2={:.4f}'.format(L))
                print('L2={:.4f}'.format(L))

            self.gc2.load_state_dict(self.gc3.state_dict())
            self.linear_a2.load_state_dict(self.linear_a3.state_dict())


            with torch.no_grad():
                mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
                assert F.mse_loss(mu, logvar) == 0
                assert F.mse_loss(mu_a, logvar_a) == 0
                Z = mu.data.numpy()


            gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')

            pre = gmm.fit_predict(Z)
            print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            self.pi_.data = torch.from_numpy(gmm.weights_).float()
            self.mu_c.data = torch.from_numpy(gmm.means_).float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).float())

            torch.save(self.state_dict(), './pretrain_model_{}.pk'.format(self.args.dataset))
        else:
            self.load_state_dict(torch.load('./pretrain_model_{}.pk'.format(self.args.dataset)))

    def predict(self,mu, logvar):
        # z_mu, z_sigma2_log, z_ma,z_a_sigma2_log = self.encoder(x,adj)
        # mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
        # z = torch.randn_like(mu) * torch.exp(z_sigma2_log / 2) + z_mu
        det=1e-10
        z  = self.reparameterize(mu,logvar)
        pi = self.pi_
        # log_sigma2_c = self.log_sigma2_c
        # mu_c = self.mu_c
        # gamma_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))
        gamma_c = torch.exp(self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
        print('gamma_c:',gamma_c)
        gamma_c=gamma_c/(gamma_c.sum(1).view(-1,1))#batch_size*Clusters
        gamma_c=F.softmax(gamma_c)
        print('gamma_c,normalized:',gamma_c)
        print('gamma_c argmax:',torch.argmax(gamma_c,1))
        print('gamma_c argmax counter:',Counter(torch.argmax(gamma_c,1).tolist()))

        gamma=gamma_c.detach().cpu().numpy()
        return np.argmax(gamma,axis=1),gamma, z

    def predict_dist(self,mu, logvar):
        # z_mu, z_sigma2_log, z_ma,z_a_sigma2_log = self.encoder(x,adj)
        # mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
        # z = torch.randn_like(mu) * torch.exp(z_sigma2_log / 2) + z_mu
        z  = self.reparameterize(mu,logvar)
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        # gamma_c = torch.exp(self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        # gamma=gamma_c.detach().cpu().numpy()

        gamma=[]
        for e in range(z.shape[0]):
            temp_dist=[]
            for m in range(mu_c.shape[0]):
                temp_dist.append(F.mse_loss(z[e],mu_c[m]).data)
            gamma.append(temp_dist)

        return np.argmin(gamma,axis=1),np.array(gamma)

    def plot_tsne(self,dataset,epoch,z,true_label,desp):

        cluster_labels=set(true_label)
        print(cluster_labels)
        index_group= [np.array(true_label)==y for y in cluster_labels]
        colors = cm.tab20(range(len(index_group)))

        tsne = TSNE(n_components=2, init='pca',perplexity=50.0)
        data = torch.cat([z,self.mu_c.to('cpu')],dim=0).detach().numpy()
        zs_tsne = tsne.fit_transform(data)

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        for index,c in zip(index_group,colors):
            ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=2)

        if 'predict' in desp.split():
            for index,c in enumerate(colors):
                ax.scatter(zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 0], zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 1],marker='^',color=c,s=40)
        else:
            ax.scatter(zs_tsne[z.shape[0]:, 0], zs_tsne[z.shape[0]:, 1],marker='^',color='b',s=40)
        plt.title(desp)
        # ax.legend()
        plt.savefig("{}_{}_tsne_{}.pdf".format(dataset,epoch,desp))

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.args.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)


    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)) # np.pi*2, not square

    def check_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data,param.data.shape)
    def check_gradient(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print('grad: ',name)
                print(param.grad,param.grad.shape)
