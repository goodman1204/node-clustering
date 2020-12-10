import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=torch.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionSparse(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=torch.relu):
        super(GraphConvolutionSparse, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.issparse = True
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Linear(Module):
    """
    to embedding feature
    """

    def __init__(self,in_features,out_features,dropout=0.,act=torch.relu,bias=True,sparse_inputs=False,**kwargs):
        super(Linear,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.sparse_inputs = sparse_inputs

        self.weight= Parameter(torch.FloatTensor(in_features,out_features))

        if self.bias:
            self.weight_bias  = Parameter(torch.FloatTensor(1,out_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias:
            torch.nn.init.xavier_uniform_(self.weight_bias)

    def forward(self,input):
        if self.sparse_inputs:
            output = torch.spmm(input,self.weight)
        else:
            output = torch.mm(input,self.weight)

        if self.bias:
            output += self.weight_bias #Find the bug, self.bias should be self.weight_bias
            # output += self.bias #Find the bug, self.bias should be self.weight_bias

        return self.act(output)


class InnerProductDecoder(torch.nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class InnerDecoder(torch.nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self,dropout=0., act=torch.sigmoid,**kwargs):
        super(InnerDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def forward(self,inputs):
        z_u, z_a = inputs
        z_u = F.dropout(z_u, self.dropout, training=self.training)
        z_a = F.dropout(z_a, self.dropout,training = self.training)
        adj = self.act(torch.mm(z_u, z_u.t())) # predicted adj matrix
        features = self.act(torch.mm(z_u,z_a.t())) #predicted feature matrix
        return adj,features
