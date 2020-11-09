import os
import time
import cdt
import argparse
import pandas as pd
import numpy as np

import sys
#sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dcdi.plot import plot_adjacency
from dcdi.utils.save import dump
from dcdi.utils.metrics import edge_errors
from dcdi.data import DataManagerFile
from pc import PC


def main(opt, metrics_callback=None, plotting_callback=None, verbose=False):
    time0 = time.time()
    opt.model = "jci_pc"

    # load data
    train_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data, random_seed=opt.random_seed, intervention=True,
                                 regimes_to_ignore=opt.regimes_to_ignore)

    gt_dag = train_data.adjacency.detach().cpu().numpy()
    train_data_pd = pd.DataFrame(train_data.dataset.detach().cpu().numpy())
    regimes_pd = pd.DataFrame(train_data.regimes)

    obj = PC()
    targets = None
    if opt.knowledge == "known":
        known = True
        targets = train_data.gt_interv[:,1:].T
    elif opt.knowledge == "unknown":
        known = False
    else:
        raise ValueError("The value of knowledge is incorrect.")

    dag = obj._run_pc(train_data_pd, regimes=regimes_pd, alpha=opt.alpha,
                      indep_test=opt.indep_test, known=known, targets=targets)
    n_nodes = train_data_pd.shape[1]
    dag = dag[:n_nodes, :n_nodes]

    # Compute SHD-CPDAG and SID metrics
    shd = float(cdt.metrics.SHD(target=gt_dag, pred=dag, double_for_anticausal=False))
    sid = float(cdt.metrics.SID(target=gt_dag, pred=dag))

    sid_cpdag = cdt.metrics.SID_CPDAG(target=gt_dag, pred=dag)
    shd_cpdag = float(cdt.metrics.SHD_CPDAG(target=gt_dag, pred=dag))
    fn, fp, rev = edge_errors(dag, gt_dag)
    timing = time.time() - time0
    if verbose:
        print(f"SID: {sid}, SHD:{shd}, SHD_CPDAG:{shd_cpdag}")
        print(f"SID lower: {sid_cpdag[0]}, SID upper:{sid_cpdag[1]}")
        print(f"fn: {fn}, fp:{fp}, rev:{rev}")

    #save
    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)

    if metrics_callback is not None:
        metrics_callback(stage="jci_pc", step=0,
                         metrics={"dag": dag, "sid": sid, "shd": shd, "alpha": alpha,
                                  "shd_cpdag": shd_cpdag, "fn": fn, "fp": fp, "rev": rev}, throttle=False)

    dump(opt, opt.exp_path, 'opt')
    dump(timing, opt.exp_path, 'timing', True)

    results = f"shd: {shd},\nsid lower: {sid_cpdag[0]},\nsid upper: {sid_cpdag[1]},\nfn: {fn},\nfp: {fp},\nrev: {rev}"
    dump(results, opt.exp_path, 'results', True)

    np.save(os.path.join(opt.exp_path, "DAG"), dag)

    plot_adjacency(gt_dag, dag, opt.exp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data files')
    parser.add_argument('--i-dataset', type=str, default=None,
                        help='dataset index')
    parser.add_argument('--exp-path', type=str, default='exp',
                        help='Path to experiments')
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='Cutoff value for independence tests')
    parser.add_argument('--indep-test', type=str, default="gaussCItest",
                        help='Independence test used (gaussCItest or kernelCItest)')
    parser.add_argument('--regimes-to-ignore', nargs="+", type=int,
                        help='When loading data, will remove some regimes from data set')
    parser.add_argument('--knowledge', type=str, default="unknown",
                        help='Are intervention targets known or unknown?')

    opt = parser.parse_args()

    opt.train_samples = 1.0 # all the data, since this is constraint-based methods
    opt.test_samples = None
    opt.random_seed = 42
    opt.normalize_data = False
    main(opt, verbose=True)
