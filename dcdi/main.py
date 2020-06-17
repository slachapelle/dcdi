# coding=utf-8

import os
import argparse
import copy
import cdt

import numpy as np
import torch

from .models.learnables import LearnableModel_NonLinGaussANM
from .models.flows import DeepSigmoidalFlowModel
from .train import train
from .data import DataManagerFile
from .utils.save import load, dump
from .utils.metrics import edge_errors

def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))

def main(opt, metrics_callback=None, plotting_callback=None):
    """
    :param opt: a Bunch-like object containing hyperparameter values
    :param metrics_callback: a function of the form f(step, metrics_dict) used to log metric values during training

    """
    # Control as much randomness as possible
    torch.manual_seed(opt.random_seed)
    np.random.seed(opt.random_seed)

    if opt.lr_reinit is not None:
        assert opt.lr_schedule is None, "--lr-reinit and --lr-schedule are mutually exclusive"

    # Dump hyperparameters to disk
    dump(opt.__dict__, opt.exp_path, 'opt')

    # Initialize metric logger if needed
    if metrics_callback is None:
        metrics_callback = _print_metrics

    # adjust some default hparams
    if opt.lr_reinit is None: opt.lr_reinit = opt.lr

    # Use GPU
    if opt.gpu:
        if opt.float:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        if opt.float:
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')

    # create experiment path
    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)

    # create DataManager for training
    train_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data,
                                 random_seed=opt.random_seed,
                                 intervention=opt.intervention, dcd=opt.dcd)
    test_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed, intervention=opt.intervention, dcd=opt.dcd)
    nb_interv = train_data.nb_interv

    # create learning model and ground truth model
    if opt.model == "DCDI-G":
        model = LearnableModel_NonLinGaussANM(opt.num_vars,
                                              opt.num_layers,
                                              opt.hid_dim,
                                              nonlin=opt.nonlin,
                                              intervention=opt.intervention,
                                              intervention_type=opt.intervention_type,
                                              intervention_knowledge=opt.intervention_knowledge,
                                              nb_interv=nb_interv)
    elif opt.model == "DCDI-DSF":
        model = DeepSigmoidalFlowModel(num_vars=opt.num_vars,
                                       cond_n_layers=opt.num_layers,
                                       cond_hid_dim=opt.hid_dim,
                                       cond_nonlin=opt.nonlin,
                                       flow_n_layers=opt.flow_num_layers,
                                       flow_hid_dim=opt.flow_hid_dim,
                                       intervention=opt.intervention,
                                       intervention_type=opt.intervention_type,
                                       intervention_knowledge=opt.intervention_knowledge,
                                       nb_interv=nb_interv)
    else:
        raise ValueError("opt.model has to be in {DCDI-G, DCDI-DSF}")



    # save gt adjacency
    dump(train_data.adjacency.detach().cpu().numpy(), opt.exp_path, 'gt-adjacency')

    # train until constraint is sufficiently close to being satisfied
    if opt.train:
        train(model, train_data.adjacency.detach().cpu().numpy(),
              train_data.gt_interv, train_data, test_data, opt, metrics_callback,
              plotting_callback)
