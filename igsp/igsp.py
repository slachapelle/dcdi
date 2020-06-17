import causaldag as cd
from causaldag.structure_learning import igsp, unknown_target_igsp
from causaldag.utils.ci_tests import gauss_ci_test, MemoizedCI_Tester, gauss_ci_suffstat
from causaldag.utils.ci_tests import hsic_test, kci_test
from causaldag.utils.invariance_tests import MemoizedInvarianceTester, gauss_invariance_test, gauss_invariance_suffstat
from causaldag.utils.invariance_tests import hsic, kci, hsic_invariance_test, kci_invariance_test
import numpy as np
from pprint import pprint
import random
import os

def format_to_igsp(data, target_raw, intervention_knowledge=False):
    """
    From the data and target, split the dataset by settings
    Return:
        obs_samples (np array): the observational samples
        iv_samples_list (list): a list of np array for each
                                interventional setting
        target_index (np array): contains the interv target
    """
    data = data.values

    nodes = set(range(data.shape[1]))
    targets = np.zeros(target_raw.shape[0])
    target_raw += 1
    iv_samples_list = []

    # replace the nan
    target_raw = np.nan_to_num(target_raw)
    target_index = np.unique(target_raw, axis=0)

    # fill targets with the target index
    for i, target in enumerate(target_raw):
        for j in range(target_index.shape[0]):
            if np.array_equal(target_index[j], target):
                targets[i] = j

    # split the dataset for each setting
    for j in range(target_index.shape[0]):
        setting = data[targets == j]
        if np.sum(target_index[j]) == 0:
            # observational case
            obs_samples = setting
        else:
            iv_samples_list.append(setting)

    target_index = np.delete(target_index, 0, 0)
    targets_list = []
    for i in range(target_index.shape[0]):
        tmp_list = []
        for t in target_index[i]:
            if t != 0:
                tmp_list.append(int(t)-1)
        targets_list.append(tmp_list)

    data2, target_raw2 = reconstruct(nodes, obs_samples, iv_samples_list, targets_list)

    return nodes, obs_samples, iv_samples_list, targets_list


def reconstruct(nodes, obs_samples, iv_samples_list, targets_list):
    data = np.concatenate(iv_samples_list, axis=0)
    data = np.concatenate([obs_samples, data], axis=0)
    target_raw = targets_list[0] * obs_samples.shape[0]
    for i in range(0, len(iv_samples_list)):
        target_raw.extend(targets_list[i] * iv_samples_list[i].shape[0])

    return data, target_raw


def prepare_igsp(obs_samples, iv_samples_list, targets_list,
                 alpha=1e-3, alpha_inv=1e-3, ci_test="gaussian"):

    # Form sufficient statistics
    if ci_test == "gaussian":
        obs_suffstat = gauss_ci_suffstat(obs_samples)
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)

        # Create CI and invariance
        ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)
    elif ci_test == "hsic":
        contexts = {i:s for i,s in enumerate(iv_samples_list)}
        invariance_suffstat = {"obs_samples":obs_samples}
        invariance_suffstat.update(contexts)

        # Create CI and invariance
        ci_tester = MemoizedCI_Tester(hsic_test, obs_samples, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(hsic_invariance_test, invariance_suffstat, alpha=alpha_inv)
    elif ci_test == "kci":
        contexts = {i:s for i,s in enumerate(iv_samples_list)}
        invariance_suffstat = {"obs_samples":obs_samples}
        invariance_suffstat.update(contexts)

        # Create CI and invariance
        ci_tester = MemoizedCI_Tester(kci_test, obs_samples, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(kci_invariance_test, invariance_suffstat, alpha=alpha_inv)
    else:
        raise ValueError(f"CI test '{ci_test}' does not exist. Choose between: [gaussian, hsic, kci]")
    return ci_tester, invariance_tester


def run_igsp(data, targets, alpha=1e-3, alpha_inv=1e-3, ci_test="gaussian"):
    """
    Apply the IGSP method.
    Return:
        est_dag: the estimated dag
    """

    nodes, obs_samples, iv_samples_list, targets_list = format_to_igsp(data, targets)
    ci_tester, invariance_tester = prepare_igsp(obs_samples,
                                                iv_samples_list, targets_list,
                                                alpha, alpha_inv, ci_test)

    # Run IGSP
    setting_list = [dict(interventions=targets) for targets in targets_list]
    est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester, nruns=5)
    dag = est_dag.to_amat()[0]

    return dag, est_dag, setting_list


def run_ut_igsp(data, targets, alpha=1e-3, alpha_inv=1e-3, ci_test="gaussian"):
    """
    Apply the UT-IGSP method.
    Return:
        est_dag: the estimated dag
        est_targets_list: the estimated target
    """

    nodes, obs_samples, iv_samples_list, targets_list = format_to_igsp(data, targets)
    ci_tester, invariance_tester = prepare_igsp(obs_samples,
                                            iv_samples_list, targets_list,
                                            alpha, alpha_inv, ci_test)

    # Run UT-IGSP
    setting_list = [dict(known_interventions=targets) for targets in targets_list]
    est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
    dag = est_dag.to_amat()[0]

    return dag, est_dag, setting_list, est_targets_list

