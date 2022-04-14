import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import numpy as np

def getSoftAssignments(latent_space, cluster_centers, num_samples):
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


    distances_from_cluster_centers = torch.sum((latent_space.unsqueeze(1)- cluster_centers.unsqueeze(0))**2,2)
    qij_numerator = 1 + distances_from_cluster_centers
    qij_numerator = 1 / qij_numerator
    normalizer_q = qij_numerator.sum(dim=1).reshape((num_samples, 1))

    return qij_numerator / normalizer_q

def calculateP(Q):
    # Function to calculate the desired distribution Q^2, for more details refer to DEC paper
    f = Q.sum(dim=0)
    pij_numerator = Q * Q
    # pij_numerator = Q
    pij_numerator = pij_numerator / f
    normalizer_p = pij_numerator.sum(dim=1).reshape((Q.shape[0], 1))
    P = pij_numerator / normalizer_p
    return P

def getKLDivLossExpression(Q_expression, P_expression):
    # Loss = KL Divergence between the two distributions
    log_arg = P_expression / Q_expression
    log_exp = torch.log(log_arg)
    sum_arg = P_expression * log_exp
    loss = torch.sum(sum_arg)/Q_expression.shape[0]
    return loss


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def get_Q(z,cluster_layer, v=1):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2) / v)
    q = q.pow((v + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

cluster_centers = torch.rand(3,4)

z = torch.rand(5,4)


q = getSoftAssignments(z,cluster_centers,5)

p = calculateP(q)

kl_loss1 = getKLDivLossExpression(q,p)
print("kl_loss1:",kl_loss1)

q = get_Q(z,cluster_centers)
p = target_distribution(q)

kl_loss2 = F.kl_div(q.log(), p, reduction='batchmean')
print("kl_loss2:",kl_loss2)


def soft_assignment(batch: torch.Tensor,cluster_centers,alpha = 1) -> torch.Tensor:
    """
    Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
    for each cluster.
    :param batch: FloatTensor of [batch size, embedding dimension]
    :return: FloatTensor [batch size, number of clusters]
    """
    norm_squared = torch.sum((batch.unsqueeze(1) - cluster_centers) ** 2, 2)
    numerator = 1.0 / (1.0 + (norm_squared / alpha))
    power = float(alpha + 1) / 2
    numerator = numerator ** power
    return numerator / torch.sum(numerator, dim=1, keepdim=True)

def target_distribution(z: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (z** 2) / torch.sum(z, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

loss_function = nn.KLDivLoss(reduction='batchmean')
q = soft_assignment(z,cluster_centers)
p = target_distribution(q)

loss = loss_function(q.log(), p)
print("kl_loss3:",loss)
