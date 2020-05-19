#!./venv/bin/python3

import os
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from tensorflow.keras.models import load_model
from utils import create_logger, count_decimal_places, ms_since_1970, TOTUtils
from argparse import ArgumentParser

default_outdir = './logs/robustness'
default_dmin = 0.00001
default_dmax = 100.0
logger = create_logger('robustness', logdir=default_outdir)

def save_local_robustness_results_to_csv(results, samples, outdir=default_outdir):
    '''
    saves local robustness summary and detailed results in csv format.

    @param results (dict): robustness results dictionary ({x0: (summary_tuple, details_list), ...})
    @param outdir (string): output directory
    @param outid (string): output file id
    '''
    summary_lines = ['dneg,dpos\n']
    details_lines = ['s,dneg,dpos,pred\n']
    summary, details = results
    summary_lines.append(','.join([str(summary[0]), str(summary[1])])+'\n')
    details_lines.extend([','.join([str(s), str(detail[0]), str(detail[1]), str(samples[s][1].index(max(samples[s][1])))])+'\n' for s,detail in enumerate(details)])
    summary_file = os.path.join(outdir, 'local_summary.csv')
    details_file = os.path.join(outdir, 'local_details.csv')
    if not os.path.exists(outdir):
        os.makedirs(outdir, mode=0o755)
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
        logger.info(f'wrote summary to {summary_file}')
    with open(details_file, 'w') as f:
        f.writelines(details_lines)
        logger.info(f'wrote detils to {details_file}')

def find_local_robustness_boundaries(model_path, samples, d_min=0.001, d_max=100, multithread=False, verbose=0, save_results=False, save_samples=False, outdir=default_outdir):
    '''
    finds local robustness of network on provided samples

    @param model_path (string): h5 or pb model path
    @param samples (list): list of samples
    @param d_min (float): min distance to consider
    @param d_max (float): max distance to consider
    @param multithread (bool): perform + and - in parallel (default false)
    @param verbose (int): extra logging (0, 1, 2) (default 0)
    @param save_results (bool): save results to csv files
    @param save_samples (bool): save samples to csv file
    @param outdir (string): output directory for csv files
    @return (tuple): ((negdist, posdist), [(s0_ndist, s0_pdist), ...(sN_ndist, sN_pdist)])
    '''
    def find_distance(s, sign):
        inputs, outputs = samples[s]
        exp_cat = outputs.index(max(outputs))
        lbound, ubound = d_min, d_max
        dist = 0
        for dp in range(dplaces+1):
            prec = round(1/(10**dp), dp)
            distances = np.arange(lbound, ubound, prec)
            dsamples = [[x+(d*sign) for x in inputs] for d in distances]
            predictions = model.predict(dsamples) if len(dsamples) > 0 else []
            # find first prediction where the category changed...
            for i,pred in enumerate(predictions):
                if list(pred).index(max(pred)) != exp_cat:
                    dist = distances[i]
                    # adjust bounds for next highest precision
                    lbound = dist - prec if dist - prec > 0 else d_min
                    ubound = dist
                    if verbose > 1: logger.info(f's{s}@p{prec}: {dlabel(sign)}={dist}')
                    break
            # give up if dist is zero after first round.
            if dist == 0:
                if verbose > 0: logger.warning(f'no {dlabel(sign)} found for s{s}@p{prec}')
                break
        if verbose > 0: logger.info(f's{s} {dlabel(sign)}={dist}')
        return dist

    model = load_model(model_path)
    start = ms_since_1970()
    dplaces = count_decimal_places(d_min)
    dlabel = lambda sign: f'{"+" if sign > 0 else "-"}dist' # for logging
    details = []
    if multithread:
        pool = ThreadPool(processes=2)
        for s in range(len(samples)):
            nthread, pthread = pool.apply_async(find_distance,(s,-1)), pool.apply_async(find_distance,(s,1))
            details.append((-1*nthread.get(), pthread.get()))
    else:
        details = [(-1*find_distance(i,-1), find_distance(i,+1)) for i in range(len(samples))]
    negd = max([d for d in [r for r in zip(*details)][0] if d is not 0] or [0])
    posd = min([d for d in [r for r in zip(*details)][1] if d is not 0] or [0])
    # results tuple containing summary tuple & details list
    results = ((negd, posd), details)
    if save_results: save_local_robustness_results_to_csv(results, samples, outdir=outdir)
    if save_samples: TOTUtils.save_samples_to_csv(samples, outdir)
    return results

if __name__ == '__main__':
    '''
    Usage: python3 verification/robustness.py -m MODELPATH -d DATAPATH [-df FRAC -dmin DMIN -dmax DMAX -mt -sr -ss -sl -o OUTDIR -v V]
    '''
    parser = ArgumentParser()
    parser.add_argument('-m', '--modelpath', required=True)
    parser.add_argument('-d', '--datapath', required=True)
    parser.add_argument('-df', '--datafrac', type=float, default=1)
    parser.add_argument('-dmin', '--dmin', type=float, default=default_dmin)
    parser.add_argument('-dmax', '--dmax', type=float, default=default_dmax)
    parser.add_argument('-mt', '--multithread', action='store_true')
    parser.add_argument('-sr', '--saveresults', action='store_true')
    parser.add_argument('-ss', '--savesamples', action='store_true')
    parser.add_argument('-sl', '--savelogs', action='store_true')
    parser.add_argument('-o', '--outdir', default=default_outdir)
    parser.add_argument('-v', '--verbose', type=int, default=0)
    args = parser.parse_args()
    # configure logger
    for handler in logger.handlers[:]: logger.removeHandler(handler)  
    logger = create_logger('robustness', to_file=args.savelogs, logdir=args.outdir)
    # load % of samples, and filter out incorrect predictions
    samples = TOTUtils.filter_samples(TOTUtils.load_samples(args.datapath, frac=args.datafrac), args.modelpath)
    find_local_robustness_boundaries(args.modelpath, samples, d_min=args.dmin, d_max=args.dmax, multithread=args.multithread, verbose=args.verbose, save_results=args.saveresults, save_samples=args.savesamples, outdir=args.outdir)
