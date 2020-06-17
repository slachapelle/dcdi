import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..dag_optim import GumbelAdjacency, GumbelIntervWeight


class BaseModel(nn.Module):
    def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu",
                 intervention=False,
                 intervention_type="perfect",
                 intervention_knowledge="known",
                 nb_interv=1,
                 nb_extra_input=2):
        """
        :param num_vars: number of variables in the system
        :param num_layers: number of hidden layers
        :param hid_dim: number of hidden units per layer
        :param num_params: number of parameters per conditional *outputted by MLP*
        :param nonlin: which nonlinearity
        :param gumbel: use GumbelAdjacency
        """
        gumbel=True
        norm_prod='path'
        square_prod=False

        super(BaseModel, self).__init__()
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_params = num_params
        self.nonlin = nonlin
        self.gumbel = gumbel
        self.norm_prod = norm_prod
        self.square_prod = square_prod
        self.intervention = intervention
        self.intervention_type = intervention_type
        self.intervention_knowledge = intervention_knowledge
        self.parametric_sharing = "none"
        self.nb_interv = nb_interv

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.extra_params = []  # Those parameter might be learnable, but they do not depend on parents.
        self.nb_extra_input = nb_extra_input

        # initialize current adjacency matrix
        self.adjacency = torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars)

        if gumbel:
            self.gumbel_adjacency = GumbelAdjacency(self.num_vars)

        if intervention_knowledge == 'unknown' and intervention:
            self.gumbel_interv_w = GumbelIntervWeight(self.num_vars, self.nb_interv)
        if intervention_knowledge == 'unknown' and intervention_type == 'perfect' and intervention:
            mu = torch.zeros(self.num_vars, 1)
            sigma = torch.ones(self.num_vars, 1)
            normal_param = torch.cat((mu, sigma), 1)
            self.interv_param = nn.Parameter(normal_param)

        self.zero_weights_ratio = 0.
        self.numel_weights = 0

        # Instantiate the parameters of each layer in the model of each variable
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim

            # first layer
            if i == 0:
                in_dim = self.num_vars

            # last layer
            if i == self.num_layers:
                out_dim = self.num_params

            if intervention_type == 'imperfect':
                self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim, nb_extra_input)))
                self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, nb_extra_input)))
                self.numel_weights += self.num_vars * out_dim * in_dim * (nb_extra_input)
            else:
                self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim)))
                self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim)))
                self.numel_weights += self.num_vars * out_dim * in_dim

    def get_interv_w(self, bs, regime):
        return self.gumbel_interv_w(bs, regime)

    def forward_given_params(self, x, weights, biases, mask=None, regime=None):
        """
        :param x: batch_size x num_vars
        :param weights: list of lists. ith list contains weights for ith MLP
        :param biases: list of lists. ith list contains biases for ith MLP
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        bs = x.size(0)
        num_zero_weights = 0
        for k in range(self.num_layers + 1):
            # apply affine operator
            if k == 0:
                if self.gumbel:
                    gumbel_adj = self.gumbel_adjacency(bs)
                    adj = self.adjacency.unsqueeze(0)

                    if self.intervention and self.intervention_type == "imperfect":
                        assert mask is not None, 'Set a value to mask to use \
                        imperfect or unknown interventions'

                        if self.intervention_knowledge == "unknown":
                            assert regime is not None, 'Set a value to regime to use \
                            imperfect unknown interventions'
                            self.interv_w = self.gumbel_interv_w(bs, regime)
                            mask = self.interv_w

                        m1 = mask.unsqueeze(2)
                        m2 = 1 - m1
                        mask = torch.cat((m1, m2), 2)

                        w = torch.einsum('tijk, btk -> btij', weights[k], mask)
                        x = torch.einsum("btij, bjt, ljt, bj -> bti", w, gumbel_adj, adj, x)
                        bias = torch.einsum("btk,tik->bti", mask, biases[k])
                        x += bias
                    else:

                        if self.intervention_knowledge == "unknown":
                            assert regime is not None, 'Set a value to regime to use \
                            perfect unknown interventions'
                            self.interv_w = self.gumbel_interv_w(bs, regime)

                        x = torch.einsum("tij,bjt,ljt,bj->bti", weights[k], gumbel_adj, adj, x) + biases[k]
            else:
                if self.intervention_type == "imperfect" and self.intervention:
                    w = torch.einsum('tijk, btk -> btij', weights[k], mask)
                    x = torch.einsum("btij, btj -> bti", w, x)
                    bias = torch.einsum("btk,tik->bti", mask, biases[k])
                    x += bias
                else:
                    x = torch.einsum("tij,btj->bti", weights[k], x) + biases[k]

            # count num of zeros
            num_zero_weights += weights[k].numel() - weights[k].nonzero().size(0)

            # apply non-linearity
            if k != self.num_layers:
                x = F.leaky_relu(x) if self.nonlin == "leaky-relu" else torch.sigmoid(x)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

        return torch.unbind(x, 1)

    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        if self.gumbel:
            return self.gumbel_adjacency.get_proba() * self.adjacency
        else:
            raise ValueError("only gumbel")

    def reset_params(self):
        with torch.no_grad():
            for node in range(self.num_vars):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu'))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        params = []

        if 'w' in mode:
            weights = []
            for w in self.weights:
                weights.append(w)
            params.append(weights)
        if 'b'in mode:
            biases = []
            for j, b in enumerate(self.biases):
                biases.append(b)
            params.append(biases)

        if 'x' in mode:
            extra_params = []
            for ep in self.extra_params:
                if ep.requires_grad:
                    extra_params.append(ep)
            params.append(extra_params)

        return tuple(params)

    def set_parameters(self, params, mode="wbx"):
        """
        Will set only parameters with requires_grad == True
        :param params: tuple of parameter lists to set, the order should be coherent with `get_parameters`
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: None
        """
        with torch.no_grad():
            k = 0
            if 'w' in mode:
                for i, w in enumerate(self.weights):
                    w.copy_(params[k][i])
                k += 1

            if 'b' in mode:
                for i, b in enumerate(self.biases):
                    b.copy_(params[k][i])
                k += 1

            if 'x' in mode and len(self.extra_params) > 0:
                for i, ep in enumerate(self.extra_params):
                    if ep.requires_grad:
                        ep.copy_(params[k][i])
                k += 1

    def get_grad_norm(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True, simply get the .grad
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        grad_norm = 0

        if 'w' in mode:
            for w in self.weights:
                grad_norm += torch.sum(w.grad ** 2)

        if 'b'in mode:
            for j, b in enumerate(self.biases):
                grad_norm += torch.sum(b.grad ** 2)

        if 'x' in mode:
            for ep in self.extra_params:
                if ep.requires_grad:
                    grad_norm += torch.sum(ep.grad ** 2)

        return torch.sqrt(grad_norm)

    def save_parameters(self, exp_path, mode="wbx"):
        params = self.get_parameters(mode=mode)
        # save
        with open(os.path.join(exp_path, "params_"+mode), 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, exp_path, mode="wbx"):
        with open(os.path.join(exp_path, "params_"+mode), 'rb') as f:
            params = pickle.load(f)
        self.set_parameters(params, mode=mode)

    def get_distribution(self, density_params):
        raise NotImplementedError
