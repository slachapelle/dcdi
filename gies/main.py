import os
import time
import cdt
import argparse
import pandas as pd
import numpy as np
import networkx as nx

import sys
#sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dcdi.plot import plot_adjacency
from dcdi.utils.save import dump
from dcdi.utils.metrics import edge_errors
from dcdi.data import DataManagerFile
from gies import GIES, retrain


def main(opt, metrics_callback=None, plotting_callback=None, verbose=False):
    time0 = time.time()
    opt.model = "gies"

    # load data
    train_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=True,
                                 normalize=opt.normalize_data, random_seed=opt.random_seed, intervention=True,
                                 regimes_to_ignore=opt.regimes_to_ignore)
    test_data = DataManagerFile(opt.data_path, opt.i_dataset, opt.train_samples, opt.test_samples, train=False,
                                normalize=opt.normalize_data, mean=train_data.mean, std=train_data.std,
                                random_seed=opt.random_seed, intervention=True, regimes_to_ignore=opt.regimes_to_ignore)
    gt_dag = train_data.adjacency.detach().cpu().numpy()
    train_data_pd = pd.DataFrame(train_data.dataset.detach().cpu().numpy())
    mask_pd = pd.DataFrame(train_data.masks)

    obj = GIES()
    dag = obj._run_gies(train_data_pd, targets=mask_pd,
                        lambda_gies=opt.lambda_gies)

    # compute score on train and valid
    train_score, val_score, flag_max_iter_retrain = retrain(dag, train_data, test_data, opt.max_iter_retrain,
                                                            opt.batch_size_retrain)

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
        metrics_callback(stage="gies", step=0,
                         metrics={"train_score": train_score, "val_score": val_score, "sid": sid, "shd": shd,
                                  "shd_cpdag": shd_cpdag, "fn": fn, "fp": fp, "rev": rev,
                                  "flag_max_iter_retrain": flag_max_iter_retrain}, throttle=False)

    dump(opt, opt.exp_path, 'opt')
    dump(timing, opt.exp_path, 'timing', True)
    dump(train_score, opt.exp_path, 'train_score', True)
    dump(val_score, opt.exp_path, 'test_score', True)

    results = f"shd: {shd},\nsid lower: {sid_cpdag[0]},\nsid upper: {sid_cpdag[1]},\nfn: {fn},\nfp: {fp},\nrev: {rev}"
    dump(results, opt.exp_path, 'results', True)

    # dump(shd, opt.exp_path, 'shd', True)
    # dump(sid_cpdag[0], opt.exp_path, 'sid_lower', True)
    # dump(sid_cpdag[1], opt.exp_path, 'sid_upper', True)
    # dump(fn, opt.exp_path, 'fn', True)
    # dump(fp, opt.exp_path, 'fp', True)
    # dump(rev, opt.exp_path, 'rev', True)

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
    # lambda should be equal to log(n)/2 where n is the number of examples
    parser.add_argument('--lambda-gies', type=float, default=1,
                        help='Penalization constant used by GIES')
    parser.add_argument('--max-iter-retrain', type=int, default=int(1e5),
                        help='maximal number of iteration for retrain')
    parser.add_argument('--batch-size-retrain', type=int, default=500,
                        help='maximal number of iteration for retrain')
    parser.add_argument('--regimes-to-ignore', nargs="+", type=int,
                        help='When loading data, will remove some regimes from data set')

    opt = parser.parse_args()

    opt.train_samples = 0.8
    opt.test_samples = None
    opt.random_seed = 42
    opt.normalize_data = False
    main(opt, verbose=True)
