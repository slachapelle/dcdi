import timeit

import numpy as np
from scipy.linalg import expm
from scipy.special import comb
import torch

from .utils.gumbel import gumbel_sigmoid


class TrExpScipy(torch.autograd.Function):
    """
    autograd.Function to compute trace of an exponential of a matrix
    """

    @staticmethod
    def forward(ctx, input):
        with torch.no_grad():
            # send tensor to cpu in numpy format and compute expm using scipy
            expm_input = expm(input.detach().cpu().numpy())
            # transform back into a tensor
            expm_input = torch.as_tensor(expm_input)
            if input.is_cuda:
                # expm_input = expm_input.cuda()
                assert expm_input.is_cuda
            # save expm_input to use in backward
            ctx.save_for_backward(expm_input)

            # return the trace
            return torch.trace(expm_input)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            expm_input, = ctx.saved_tensors
            return expm_input.t() * grad_output


def compute_dag_constraint(model, w_adj):
    assert (w_adj >= 0).detach().cpu().numpy().all()
    h = TrExpScipy.apply(w_adj) - model.num_vars
    return h


def compute_01_constraint(w_adj):
    """We expect w_adj to be a matrix of probabilies: each entry in [0,1]"""
    return w_adj * (1 - w_adj)


def is_acyclic(adjacency):
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0: return False
    return True


class GumbelAdjacency(torch.nn.Module):
    def __init__(self, num_vars):
        super(GumbelAdjacency, self).__init__()
        self.num_vars = num_vars
        self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.reset_parameters()

    def forward(self, bs, tau=1, drawhard=True):
        adj = gumbel_sigmoid(self.log_alpha, self.uniform, bs, tau=tau, hard=drawhard)
        return adj

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 5)


class GumbelIntervWeight(torch.nn.Module):
    def __init__(self, num_vars, num_interv):
        super(GumbelIntervWeight, self).__init__()
        self.num_vars = num_vars
        self.num_interv = num_interv
        self.log_alpha = torch.nn.Parameter(torch.ones((num_vars, num_interv)) * 3)
        self.uniform = torch.distributions.uniform.Uniform(0, 1)

    def forward(self, bs, regime, tau=1, drawhard=True):
        regime = regime.type(torch.LongTensor)
        interv_w = gumbel_sigmoid(self.log_alpha[:,regime], self.uniform, 1, tau=tau, hard=drawhard)
        interv_w = interv_w.squeeze().transpose(0, 1)
        return interv_w

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha)
