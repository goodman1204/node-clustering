import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, m,logvar, n_nodes, n_features, norm, pos_weight):
    preds_sub_u, preds_sub_a = preds
    labels_sub_u, labels_sub_a = labels
    z_u, z_a = m
    logvar_u, logvar_a = logvar
    norm_u, norm_a = norm
    pos_weight_u, pos_weight_a = pos_weight


    cost_u = norm_u * F.binary_cross_entropy_with_logits(preds_sub_u, labels_sub_u, pos_weight = pos_weight_u)
    cost_a = norm_a * F.binary_cross_entropy_with_logits(preds_sub_a, labels_sub_a, pos_weight = pos_weight_a)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_u = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar_u - z_u.pow(2) - logvar_u.exp().pow(2), 1))

    KLD_a = -0.5 / n_features * torch.mean(torch.sum(
        1 + 2 * logvar_a - z_a.pow(2) - logvar_a.exp().pow(2), 1))
    return cost_u, cost_a, KLD_u, KLD_a
