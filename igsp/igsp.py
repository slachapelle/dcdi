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

def format_to_igsp(data, targets, regimes, intervention_knowledge=False):
    """
    From the data, targets and regimes, split the dataset by interventional targets
    in order to run the IGSP or UTIGSP method.
    Return:
        nodes (set): a set of the graph nodes
        obs_samples (nd array): the observational samples
        iv_samples_list (list): a list of nd array for each
                                interventional setting
        targets_list (list): contains the interv target
                             for each element of iv_samples_list
    """
    data = data.values
    nodes = set(range(data.shape[1]))
    targets += 1
    # replace the nan
    targets = np.nan_to_num(targets)
    n_regimes = np.unique(regimes, axis=0).shape[0]

    iv_samples_list = []
    regime2target = {}

    # populate the dict regime2target
    for i, t in enumerate(regimes):
        if t not in regime2target:
            tmp_list = []
            for t_raw in targets[i]:
                if t_raw != 0:
                    tmp_list.append(int(t_raw)-1)
            regime2target[t] = tmp_list

    # split the dataset for each setting
    targets_list = []
    for j in range(n_regimes):
        setting = data[regimes == j]
        if j == 0:
            # observational case
            obs_samples = setting
        else:
            iv_samples_list.append(setting)
            targets_list.append(regime2target[j])

    return nodes, obs_samples, iv_samples_list, targets_list


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


def run_igsp(data, targets, regimes, alpha=1e-3, alpha_inv=1e-3, ci_test="gaussian"):
    """
    Apply the IGSP method.
    Return:
        dag(nd array): the adjacency matrix of the estimated DAG
        est_dag(object): the estimated dag as a DAG object from causaldag
        setting_list(list): list containing the targets for each setting
    """

    nodes, obs_samples, iv_samples_list, targets_list = format_to_igsp(data, targets, regimes)
    ci_tester, invariance_tester = prepare_igsp(obs_samples,
                                                iv_samples_list, targets_list,
                                                alpha, alpha_inv, ci_test)

    # Run IGSP
    setting_list = [dict(interventions=targets) for targets in targets_list]
    # setting_list = dict(interventions=targets_list)
    print(setting_list)
    est_dag = igsp(setting_list, nodes, ci_tester, invariance_tester, nruns=5)
    dag = est_dag.to_amat()[0]

    return dag, est_dag, setting_list


def run_ut_igsp(data, targets, regimes, alpha=1e-3, alpha_inv=1e-3, ci_test="gaussian"):
    """
    Apply the UT-IGSP method.
    Return:
        dag(nd array): the adjacency matrix of the estimated DAG
        est_dag(object): the estimated dag as a DAG object from causaldag
        setting_list(list): list containing the targets per setting
        est_targets_list(list): list of the estimated targets per setting
    """

    nodes, obs_samples, iv_samples_list, targets_list = format_to_igsp(data, targets, regimes)
    ci_tester, invariance_tester = prepare_igsp(obs_samples,
                                            iv_samples_list, targets_list,
                                            alpha, alpha_inv, ci_test)

    # Run UT-IGSP
    setting_list = [dict(known_interventions=targets) for targets in targets_list]
    est_dag, est_targets_list = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
    dag = est_dag.to_amat()[0]

    return dag, est_dag, setting_list, est_targets_list
