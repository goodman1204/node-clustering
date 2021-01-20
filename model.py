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
from utils import cluster_acc,clustering_evaluation

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

		KLD = -0.5 / n_nodes * torch.mean(torch.sum(-1 - 2 * logvar + mu.pow(2) + logvar.exp().pow(2),1))
		return L_rec_u, -KLD


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

	def pre_train(self,x,adj,Y,pre_epoch=50):
		'''
		This function is used to initialize  cluster paramters: pi_, mu_c, log_sigma2_c.
		-------------
		paramters:
		x: is the feature matrix of graph G.
		adj: is the adjacent matrix of graph G.
		Y: is the class label for each node in graph G.
		'''

		if	not os.path.exists('./pretrain_model_{}.pk'.format(self.args.dataset)):

			Loss=nn.MSELoss()
			opti=Adam(self.parameters()) #all paramters in model

			print('Pretraining......')
			# epoch_bar=tqdm(range(pre_epoch))
			# for _ in epoch_bar:
			for _ in range(pre_epoch):

				self.train()
				L=0
				mu, logvar	= self.encoder(x,adj)
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
				mu, logvar	= self.encoder(x,adj)
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
		self.modularity_layer= Linear(hidden_dim2,args.nClusters,act=torch.relu)
		# self.cluster_choose= Linear(hidden_dim2,args.nClusters,act=torch.relu)

		self.pi_=nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1)/args.nClusters,requires_grad=True)
		self.mu_c=nn.Parameter(torch.FloatTensor(args.nClusters,hidden_dim2).fill_(0.00),requires_grad=True)
		self.log_sigma2_c=nn.Parameter(torch.FloatTensor(args.nClusters,hidden_dim2).fill_(0.0),requires_grad=True)

		torch.nn.init.xavier_normal_(self.mu_c)
		torch.nn.init.xavier_normal_(self.log_sigma2_c)

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

		# adj = adj.to_dense()
		H = self.modularity_layer(z)
		assert H.shape[0]==z.shape[0]

		n = torch.tensor(1.0*z.shape[0])

		H_norm = n.sqrt()*H.sqrt()/(H.sqrt().sum())
		# print("H_norm shape",H_norm.shape)
		# print("H_norm ",H_norm)
		m = (adj-torch.eye(adj.shape[0]).cuda()).sum()/2
		D = (adj-torch.eye(adj.shape[0]).cuda()).sum(1) # the degree of nodes, adj includes self loop
		B = (adj-torch.eye(adj.shape[0]).cuda())-torch.matmul(D.view(-1,1),D.view(1,-1))/(2*m) # modularity matrix
		mod_loss=torch.trace(torch.matmul(torch.matmul(H_norm.t(),B),H_norm)/(4*m))
		# print("mod_loss",mod_loss)

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

		# KLD_a = (0.5 / n_features) * torch.mean(torch.sum(-1 - 2 * logvar_a + mu_a.pow(2) + logvar_a.exp().pow(2), 1))
		KLD_a = -(0.5 / n_features) * torch.mean(torch.sum(-1 - 2 * logvar_a + mu_a.pow(2) + logvar_a.exp().pow(2), 1))
		# KLD_a =torch.Tensor(1).fill_(0)

		# Loss=L_rec*x.size(1)


		# log_sigma2_c=self.log_sigma2_c
		# mu_c=self.mu_c

		# z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
		z = self.reparameterize(mu,logvar)

		# how about fusing attribute embeddings into node embedding?

		z_a = self.reparameterize(mu_a,logvar_a)

		H = torch.matmul(x,z_a)
		assert H.shape[0],H.shape[1] == (n_nodes,self.args.hidden2)


		# mod_loss=self.modularity_loss(z,adj)
		# gamma_c=torch.exp(torch.log(self.pi_.unsqueeze(0))+self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
		# gamma_c=torch.exp(self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
		# gamma_c  = self.cluster_choose(self.reparameterize(mu,logvar))
		# print('gamma_c:',gamma_c)

		# gamma_c=gamma_c/(gamma_c.sum(1).view(-1,1))#batch_size*Clusters
		# gamma_c=F.softmax(gamma_c)
		# print('gamma_c normalized:',gamma_c)
		# print('gamma_c argmax:',torch.argmax(gamma_c,1))
		# print('gamma_c counter:',Counter(torch.argmax(gamma_c,1).tolist()))

		# gamma_c=torch.nn.functional.one_hot(torch.argmax(gamma_c,1),self.args.nClusters)

		# self.pi_.data = (self.pi_/self.pi_.sum()).data # prior need to be re-normalized? In GMM, prior is based on gamma_c:https://brilliant.org/wiki/gaussian-mixture-model/
		# self.pi_.data = gamma_c.mean(0).data # prior need to be re-normalized? In GMM, prior is based on gamma_c:https://brilliant.org/wiki/gaussian-mixture-model/

		# multiple gaussian priors for
		# KLD_u_c=-(0.5/n_nodes)*torch.mean(torch.sum(gamma_c*torch.sum(-1+self.log_sigma2_c.unsqueeze(0)-2*logvar.unsqueeze(1)+torch.exp(2*logvar.unsqueeze(1)-self.log_sigma2_c.unsqueeze(0))+(mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.log_sigma2_c.unsqueeze(0)),2),1))
		#single KLD_u
		KLD_u_c= -0.5 / n_nodes * torch.mean(torch.sum(-1 - 2 * logvar + mu.pow(2) + logvar.exp().pow(2),1))

		# KLD_u_c=-(0.5/n_nodes)*torch.mean(torch.sum(gamma_c*torch.sum(-1-2*logvar.unsqueeze(1)+torch.exp(2*logvar.unsqueeze(1))+(mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2),2),1))
		# temp_kld=-(0.5/n_nodes)*torch.sum((mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2),2)

		# KLD_u_c_test=-(0.5/n_nodes)*F.mse_loss(mu.unsqueeze(1),self.mu_c.unsqueeze(0),reduction='none')
		# print('kld_u_c_test:',KLD_u_c_test.sum(2))


		# KLD_u_c=-(0.5/n_nodes)*F.mse_loss(mu.unsqueeze(1),self.mu_c.unsqueeze(0))

		# KLD_u_c=(0.5 / n_nodes)*torch.mean(torch.sum(gamma_c*torch.sum(self.log_sigma2_c.unsqueeze(0)+\
			# torch.exp(2*logvar.unsqueeze(1)-self.log_sigma2_c.unsqueeze(0))+\
			# (mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.log_sigma2_c.unsqueeze(0)),2),1))

		mutual_dist = (1/(self.args.nClusters**2))*self.dist(self.mu_c)

		# gamma_loss=-(1/self.args.nClusters)*torch.mean(torch.sum(gamma_c*torch.log(gamma_c),1))
		# gamma_loss = (1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c),1)) - (0.5 / self.args.hid_dim)*torch.mean(torch.sum(1+2*logvar,1))
		# gamma_loss = -(1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c/self.pi_.unsqueeze(0)),1))
		# gamma_loss = (1 / self.args.nClusters) * torch.mean(torch.sum(gamma_c*torch.log(gamma_c/self.pi_.unsqueeze(0)),1)) - (0.5 / self.args.hid_dim)*torch.mean(torch.sum(1+2*logvar,1))

		# soft cluster assignment

		# z = torch.cat((z,H),dim=1)
		# z = z+H

		print('z shape mu_c shape',z.shape,self.mu_c.shape)
		Q = self.getSoftAssignments(z,self.mu_c.cuda(),n_nodes)

		P = self.calculateP(Q)
		# if epoch ==0:
			# P = self.calculateP(Q)

		# if epoch!=0 and epoch%5==0:
			# P = self.calculateP(Q)

		soft_cluster_loss = self.getKLDivLossExpression(Q,P)

		print("Soft cluster assignment",Counter(torch.argmax(Q,1).tolist()))

		# return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a
		# return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a , -0.1*soft_cluster_loss
		# return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a , -0.1*mutual_dist,-0.01*soft_cluster_loss
		return [L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a,-0.02*soft_cluster_loss],[mu,logvar,mu_a,logvar_a,z]
		# return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a , -gamma_loss, -0.1*soft_cluster_loss
		# return L_rec_u , L_rec_a , -KLD_u_c ,-KLD_a , -gamma_loss,-mi_a
		# return L_rec_u + L_rec_a + KLD_u_c + KLD_a + gamma_loss


	def pre_train(self,x,adj,Y,pre_epoch=22):
		'''
		This function is used to initialize  cluster paramters: pi_, mu_c, log_sigma2_c.
		-------------
		paramters:
		x: is the feature matrix of graph G.
		adj: is the adjacent matrix of graph G.
		Y: is the class label for each node in graph G.
		'''

		if not os.path.exists('./pretrain_model_{}_{}.pk'.format(self.args.dataset,pre_epoch)):

			Loss=nn.MSELoss()
			opti=Adam(self.parameters()) #all paramters in model

			print('Pretraining......')
			# epoch_bar=tqdm(range(pre_epoch))
			# for _ in epoch_bar:
			for _ in range(pre_epoch):

				self.train()
				L=0
				mu, logvar, mu_a, logvar_a	= self.encoder(x,adj)
				pred_adj, pred_x = self.decoder(mu,mu_a,logvar,logvar_a)

				loss=  F.mse_loss(pred_x,x) + F.mse_loss(pred_adj,adj)

				# L+=loss

				opti.zero_grad()
				loss.backward()
				opti.step()

				# epoch_bar.write('L2={:.4f}'.format(L))
				print('L2={:.4f}'.format(loss.item()))

			# self.gc2.load_state_dict(self.gc3.state_dict())
			# self.linear_a2.load_state_dict(self.linear_a3.state_dict())


			# with torch.no_grad():
				# mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
				# assert F.mse_loss(mu, logvar) == 0
				# assert F.mse_loss(mu_a, logvar_a) == 0
				# Z = mu.data.numpy()

			mu, logvar, mu_a, logvar_a	= self.encoder(x,adj)
			Z  = self.reparameterize(mu,logvar)

			gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')

			pre = gmm.fit_predict(Z.cpu().detach().numpy())
			# print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))
			H, C, V, ari, ami, nmi, purity	= clustering_evaluation(pre,Y)
			print("purity, NMI:",purity,nmi)
			self.plot_tsne(self.args.dataset,pre_epoch,Z.to('cpu'),Y,pre)

			self.pi_= torch.nn.Parameter(torch.from_numpy(gmm.weights_))
			self.mu_c = torch.nn.Parameter(torch.from_numpy(gmm.means_))
			self.log_sigma2_c = torch.nn.Parameter(torch.from_numpy(gmm.covariances_))

			torch.save(self.state_dict(), './pretrain_model_{}_{}.pk'.format(self.args.dataset,pre_epoch))
		else:
			self.load_state_dict(torch.load('./pretrain_model_{}_{}.pk'.format(self.args.dataset,pre_epoch)))


	# def predict_nn(self,mu,logvar):
		# z  = self.reparameterize(mu,logvar)
		# gamma_c  = self.cluster_choose(self.reparameterize(mu,logvar))

		# print('gamma_c,normalized:',gamma_c)
		# print('gamma_c argmax:',torch.argmax(gamma_c,1))
		# print('gamma_c argmax counter:',Counter(torch.argmax(gamma_c,1).tolist()))

		# gamma=gamma_c.detach().cpu().numpy()


		# return np.argmax(gamma,axis=1),gamma, z


	def predict_soft_assignment(self, mu, logvar,z):

		# z_mu, z_sigma2_log, z_ma,z_a_sigma2_log = self.encoder(x,adj)
		# mu, logvar, mu_a, logvar_a  = self.encoder(x,adj)
		# z = torch.randn_like(mu) * torch.exp(z_sigma2_log / 2) + z_mu
		det=1e-10
		# z  = self.reparameterize(mu,logvar)
		Q = self.getSoftAssignments(z,self.mu_c.cuda(),z.shape[0])

		pi = self.pi_
		# log_sigma2_c = self.log_sigma2_c
		# mu_c = self.mu_c
		# gamma_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))
		# gamma_c = torch.exp(self.gaussian_pdfs_log(mu,self.mu_c,self.log_sigma2_c))+det
		# gamma_c = torch.exp(self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
		gamma_c = Q
		# print('gamma_c:',gamma_c)
		# gamma_c=gamma_c/(gamma_c.sum(1).view(-1,1))#batch_size*Clusters
		# gamma_c=F.softmax(gamma_c)
		# print('gamma_c,normalized:',gamma_c)
		# print('gamma_c argmax:',torch.argmax(gamma_c,1))

		gamma=gamma_c.detach().cpu().numpy()


		return np.argmax(gamma,axis=1),gamma, z

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
		gamma_c = torch.exp(self.gaussian_pdfs_log(mu,self.mu_c,self.log_sigma2_c))+det
		# gamma_c = torch.exp(self.gaussian_pdfs_log(z,self.mu_c,self.log_sigma2_c))+det
		print('gamma_c:',gamma_c)
		gamma_c=gamma_c/(gamma_c.sum(1).view(-1,1))#batch_size*Clusters
		# gamma_c=F.softmax(gamma_c)
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

	def plot_tsne(self,dataset,epoch,z,true_label,pred_label):

		tsne = TSNE(n_components=2, init='pca',perplexity=50.0)
		data = torch.cat([z,self.mu_c.to('cpu').float()],dim=0).detach().numpy()
		zs_tsne = tsne.fit_transform(data)

		cluster_labels=set(true_label)
		print(cluster_labels)
		index_group= [np.array(true_label)==y for y in cluster_labels]
		colors = cm.tab20(range(len(index_group)))

		fig, ax = plt.subplots()
		for index,c in zip(index_group,colors):
			ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=2)

		ax.scatter(zs_tsne[z.shape[0]:, 0], zs_tsne[z.shape[0]:, 1],marker='^',color='b',s=40)
		plt.title('true label')
		# ax.legend()
		plt.savefig("./visualization/{}_{}_tsne_{}.pdf".format(dataset,epoch,'true_label'))

		cluster_labels=set(pred_label)
		print(cluster_labels)
		index_group= [np.array(pred_label)==y for y in cluster_labels]
		colors = cm.tab10(range(len(index_group)))

		fig, ax = plt.subplots()
		for index,c in zip(index_group,colors):
			ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=2)

		for index,c in enumerate(colors):
			ax.scatter(zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 0], zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 1],marker='^',color=c,s=40)

		plt.title('pred label')
		# ax.legend()
		plt.savefig("./visualization/{}_{}_tsne_{}.pdf".format(dataset,epoch,'pred_label'))

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

	def calculateP(self, Q):
		# Function to calculate the desired distribution Q^2, for more details refer to DEC paper
		f = Q.sum(dim=0)
		pij_numerator = Q * Q
		# pij_numerator = Q
		pij_numerator = pij_numerator / f
		normalizer_p = pij_numerator.sum(dim=1).reshape((Q.shape[0], 1))
		P = pij_numerator / normalizer_p
		return P

	def getKLDivLossExpression(self, Q_expression, P_expression):
		# Loss = KL Divergence between the two distributions
		log_arg = P_expression / Q_expression
		log_exp = torch.log(log_arg)
		sum_arg = P_expression * log_exp
		loss = torch.sum(sum_arg)
		return loss

	def getSoftAssignments(self,latent_space, cluster_centers, num_samples):
		'''
		Returns cluster membership distribution for each sample
		:param latent_space: latent space representation of inputs
		:param cluster_centers: the coordinates of cluster centers in latent space
		:param num_clusters: total number of clusters
		:param latent_space_dim: dimensionality of latent space
		:param num_samples: total number of input samples
		:return: soft assigment based on the equation qij = (1+|zi - uj|^2)^(-1)/sum_j'((1+|zi - uj'|^2)^(-1))
		'''
		# z_expanded = latent_space.reshape((num_samples, 1, latent_space_dim))
		# z_expanded = T.tile(z_expanded, (1, num_clusters, 1))
		# u_expanded = T.tile(cluster_centers, (num_samples, 1, 1))

		# distances_from_cluster_centers = (z_expanded - u_expanded).norm(2, axis=2)
		# qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
		# qij_numerator = 1 / qij_numerator
		# normalizer_q = qij_numerator.sum(axis=1).reshape((num_samples, 1))

		# return qij_numerator / normalizer_q


		distances_from_cluster_centers = (latent_space.unsqueeze(1)- cluster_centers.unsqueeze(0)).norm(2, dim=2)
		qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
		qij_numerator = 1 / qij_numerator
		normalizer_q = qij_numerator.sum(dim=1).reshape((num_samples, 1))

		return qij_numerator / normalizer_q

	def init_clustering_params(self,gmm):

		self.pi_= torch.nn.Parameter(torch.from_numpy(gmm.weights_))
		self.mu_c = torch.nn.Parameter(torch.from_numpy(gmm.means_))
		self.log_sigma2_c = torch.nn.Parameter(torch.from_numpy(gmm.covariances_))
		print(self.mu_c)

	def init_clustering_params_kmeans(self,km):

		# self.pi_= torch.nn.Parameter(torch.from_numpy(gmm.weights_))
		self.mu_c = torch.nn.Parameter(torch.from_numpy(km.cluster_centers_))
		# self.log_sigma2_c = torch.nn.Parameter(torch.from_numpy(gmm.covariances_))
		# print(self.mu_c)
		# self.pi_=self.pi_.to('cuda')
		# self.mu_c=self.mu_c.to('cuda')
		# self.log_sigma2_c= self.log_sigma2_c.to('cuda')
