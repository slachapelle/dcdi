# coding=utf-8
"""
GraN-DAG

Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import os
import argparse
import cdt
import torch
import numpy as np

from .models.learnables import LearnableModel_NonLinGaussANM
from .models.flows import DeepSigmoidalFlowModel
from .train import train, retrain, compute_loss
from .data import DataManagerFile
from .utils.save import dump

def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))

def main(opt, metrics_callback=_print_metrics, plotting_callback=None):
    """
    :param opt: a Bunch-like object containing hyperparameter values
    :param metrics_callback: a function of the form f(step, metrics_dict)
        used to log metric values during training
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

    # raise error if not valid setting
    if not(not opt.intervention or \
    (opt.intervention and opt.intervention_type == "perfect" and opt.intervention_knowledge == "known") or \
    (opt.intervention and opt.intervention_type == "perfect" and opt.intervention_knowledge == "unknown") or \
    (opt.intervention and opt.intervention_type == "imperfect" and opt.intervention_knowledge == "known")):
        raise ValueError("Not implemented")

    # if observational, force interv_type to perfect/known
    if not opt.intervention:
        print("No intervention")
        opt.intervention_type = "perfect"
        opt.intervention_knowledge = "known"

    # create DataManager for training
    train_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data,
                                 random_seed=opt.random_seed,
                                 intervention=opt.intervention,
                                 intervention_knowledge=opt.intervention_knowledge,
                                 dcd=opt.dcd,
                                 regimes_to_ignore=opt.regimes_to_ignore)
    test_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed,
                                intervention=opt.intervention,
                                intervention_knowledge=opt.intervention_knowledge,
                                dcd=opt.dcd,
                                regimes_to_ignore=opt.regimes_to_ignore)

    # create learning model and ground truth model
    if opt.model == "DCDI-G":
        model = LearnableModel_NonLinGaussANM(opt.num_vars,
                                              opt.num_layers,
                                              opt.hid_dim,
                                              nonlin=opt.nonlin,
                                              intervention=opt.intervention,
                                              intervention_type=opt.intervention_type,
                                              intervention_knowledge=opt.intervention_knowledge,
                                              num_regimes=train_data.num_regimes)
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
                                       num_regimes=train_data.num_regimes)
    else:
        raise ValueError("opt.model has to be in {DCDI-G, DCDI-DSF}")


    # save gt adjacency
    dump(train_data.adjacency.detach().cpu().numpy(), opt.exp_path, 'gt-adjacency')

    # train until constraint is sufficiently close to being satisfied
    if opt.train:
        train(model, train_data.adjacency.detach().cpu().numpy(),
              train_data.gt_interv, train_data, test_data, opt, metrics_callback,
              plotting_callback)

    elif opt.retrain:
        initial_dag = np.load(opt.dag_for_retrain)
        model.adjacency[:, :] = torch.as_tensor(initial_dag).type(torch.Tensor)
        best_model = retrain(model, train_data, test_data, "ignored_regimes", opt, metrics_callback, plotting_callback)

    # Evaluate on ignored regimes!
    if opt.test_on_new_regimes:
        all_regimes = train_data.all_regimes

        # take all data, but ignore data on which we trained (want to test on unseen regime)
        regimes_to_ignore = np.setdiff1d(all_regimes, np.array(opt.regimes_to_ignore))
        new_data = DataManagerFile(opt.data_path, opt.i_dataset, 1., None, train=True,
                                   normalize=opt.normalize_data,
                                   random_seed=opt.random_seed,
                                   intervention=opt.intervention,
                                   intervention_knowledge=opt.intervention_knowledge,
                                   dcd=opt.dcd,
                                   regimes_to_ignore=regimes_to_ignore)

        with torch.no_grad():
            weights, biases, extra_params = best_model.get_parameters(mode="wbx")

            # evaluate on train
            x, masks, regimes = train_data.sample(train_data.num_samples)
            loss_train, mean_std_train = compute_loss(x, masks, regimes, best_model, weights, biases, extra_params,
                                                  intervention=True, intervention_type='structural',
                                                  intervention_knowledge="known", mean_std=True)

            # evaluate on valid
            x, masks, regimes = test_data.sample(test_data.num_samples)
            loss_test, mean_std_test = compute_loss(x, masks, regimes, best_model, weights, biases, extra_params,
                                                    intervention=True, intervention_type='structural',
                                                    intervention_knowledge="known", mean_std=True)

            # evaluate on new intervention
            x, masks, regimes = new_data.sample(new_data.num_samples)
            loss_new, mean_std_new = compute_loss(x, masks, regimes, best_model, weights, biases, extra_params,
                                                  intervention=True, intervention_type='structural',
                                                  intervention_knowledge="known", mean_std=True)

            # logging final result
            metrics_callback(stage="test_on_new_regimes", step=0,
                             metrics={"log_likelihood_train": - loss_train.item(),
                                      "mean_std_train": mean_std_train.item(),
                                      "log_likelihood_test": - loss_test.item(),
                                      "mean_std_test": mean_std_test.item(),
                                      "log_likelihood_new": - loss_new.item(),
                                      "mean_std_new": mean_std_new.item()}, throttle=False)
