#!/usr/bin/python3

import os
import numpy as np
from multiprocessing.pool import ThreadPool
from tensorflow.keras.models import load_model
from utils import create_logger, count_decimal_places, ms_since_1970, TOTUtils
from maraboupy import Marabou
from argparse import ArgumentParser

default_outdir = './logs/sensitivity'
default_dmin = 0.00001
default_dmax = 100.0
logger = create_logger('sensitivity', logdir=default_outdir)

def evaluate_sample(nnet_path, inputs, outputs):
    '''
    evaluates a sample using marabou

    @param nnet_path (string): path to the nnet file
    @param inputs (list): x values for sample
    @param outputs (list): y values for sample
    @return (tuple): (UNSAT|SAT, [y0,...,yN])
    '''
    expected_cat = outputs.index(max(outputs))
    other_cats = [i for i in range(len(outputs)) if i != expected_cat]
    net = Marabou.read_nnet(nnet_path)
    for x,v in enumerate(inputs):
        net.setLowerBound(net.inputVars[0][x], v)
        net.setUpperBound(net.inputVars[0][x], v)
    pred = net.evaluate([inputs])
    pred_cat = list(pred[0]).index(max(pred[0]))
    result = 'UNSAT' if pred_cat == expected_cat else 'SAT'
    return (result, pred)

def save_sensitivity_results_to_csv(results, outdir=default_outdir):
    '''
    saves sensitivity summary and detailed results in csv format.

    @param results (dict): sensitivity results dictionary ({x0: (summary_tuple, details_list), ...})
    @param outdir (string): output directory
    '''
    summary_lines = ['x,dneg,dpos\n']
    details_lines = ['x,s,dneg,dpos\n']
    for x_id, result in results.items():
        x = int(x_id[1:])
        summary, details = result
        summary_lines.append(','.join([str(x), str(summary[0]), str(summary[1])])+'\n')
        details_lines.extend([','.join([str(x), str(s), str(detail[0]), str(detail[1])])+'\n' for s,detail in enumerate(details)])
    summary_file = os.path.join(outdir, 'summary.csv')
    details_file = os.path.join(outdir, 'details.csv')
    if not os.path.exists(outdir):
        os.makedirs(outdir, mode=0o755)
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
        logger.info(f'wrote summary to {summary_file}')
    with open(details_file, 'w') as f:
        f.writelines(details_lines)
        logger.info(f'wrote detils to {details_file}')

def find_feature_sensitivity_boundaries(model_path, x, samples, d_min=default_dmin, d_max=default_dmax, multithread=False, verbose=0):
    '''
    finds +/- sensitivity boundaries of a feature on a set of input samples (note predictions must be correct)
    
    @param model_path (string): h5 or pb model path
    @param x (int): index of feature (x0 to xN)
    @param samples (list): list of tuples containing input and output samples
    @d_min (float): min distance to consider (also used as precision) (default false)
    @d_max (float): max distance to consider (default false)
    @multithread (bool): perform + and - in parallel (default false)
    @verbose (int): extra logging (0, 1, 2) (default 0)
    @return (tuple): ((negdist,posdist), [(s0_ndist,s0_pdist)...(sN_ndist,sN_pdist)])
    '''
    def find_distance(s, sign):
        inputs, outputs = samples[s]
        exp_cat = outputs.index(max(outputs))
        lbound, ubound = d_min, d_max
        dist = 0
        for dp in range(dplaces+1):
            prec = round(1/(10**dp), dp)
            distances = np.arange(lbound, ubound, prec)
            adj_s = [inputs[0:x]+[inputs[x]+(d*sign)]+inputs[x+1:] for d in distances]
            predictions = model.predict(adj_s) if len(adj_s) > 0 else []
            # find first prediction where the category changed...
            for i,pred in enumerate(predictions):
                if list(pred).index(max(pred)) != exp_cat:
                    dist = distances[i]
                    # adjust bounds for next highest precision
                    lbound = dist - prec if dist - prec > 0 else d_min
                    ubound = dist
                    if verbose > 1: logger.info(f'x{x}_s{s}@p{prec}: {dlabel(sign)}={dist}')
                    break
            # give up if dist is zero after first round.
            if dist == 0:
                if verbose > 0: logger.warning(f'no {dlabel(sign)} found for x{x}_s{s}@p{prec}')
                break
        if verbose > 0: logger.info(f'x{x}_s{s} {dlabel(sign)}={dist}')
        return dist

    model = load_model(model_path)
    start = ms_since_1970()
    dplaces = count_decimal_places(d_min)
    dlabel = lambda sign: f'{"+" if sign > 0 else "-"}dist' # for logging
    results = []
    if multithread:
        pool = ThreadPool(processes=2)
        for i in range(len(samples)):
            nthread, pthread = pool.apply_async(find_distance,(i,-1)), pool.apply_async(find_distance,(i,1))
            results.append((-1*nthread.get(), pthread.get()))
    else:
        results = [(-1*find_distance(i,-1), find_distance(i,+1)) for i in range(len(samples))]
    negd = max([d for d in [r for r in zip(*results)][0] if d is not 0] or [0])
    posd = min([d for d in [r for r in zip(*results)][1] if d is not 0] or [0])
    if verbose > 0: logger.info(f'x{x}: ({negd}, {posd}) ({ms_since_1970() - start}ms)')
    return ((negd, posd), results)

def find_sensitivity_boundaries(model_path, samples, d_min=default_dmin, d_max=default_dmax, multithread=False, verbose=0, save_results=False, save_samples=False, outdir=default_outdir):
    '''
    finds sensitivity for all features on provided samples

    @param model_path (string): h5 or pb model path
    @param samples (list): list of samples
    @param d_min (float): min distance to consider
    @param d_max (float): max distance to consider
    @param multithread (bool): perform + and - in parallel (default false)
    @param verbose (int): extra logging (0, 1, 2) (default 0)
    @param save_results (bool): save results to csv files
    @param save_samples (bool): save samples to csv file
    @param outdir (string): output directory for csv files
    @return (dict): {x0:((negdist,posdist), [(x0s0_ndist,x0s1_pdist)...(xNsM_ndist,xNsM_pdist)]),...}
    '''
    n_features = len(samples[0][0])
    results = {}
    for x in range(n_features):
        results[f'x{x}'] = find_feature_sensitivity_boundaries(model_path, x, samples, d_min=d_min, d_max=d_max, multithread=multithread, verbose=verbose)
    if save_results: save_sensitivity_results_to_csv(results, outdir=outdir)
    if save_samples: TOTUtils.save_samples_to_csv(samples, outdir)
    return results

def main():
    '''
    Usage: python3 verification/sensitivity.py -m MODELPATH -d DATAPATH [-df FRAC -dmin DMIN -dmax DMAX -mt -sr -ss -sl -o OUTDIR -v V]
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
    logger = create_logger('sensitivity', to_file=args.savelogs, logdir=args.outdir)
    # load % of samples, and filter out incorrect predictions
    samples = TOTUtils.filter_samples(TOTUtils.load_samples(args.datapath, frac=args.datafrac), args.modelpath)
    find_sensitivity_boundaries(args.modelpath, samples, d_min=args.dmin, d_max=args.dmax, multithread=args.multithread, verbose=args.verbose, save_results=args.saveresults, save_samples=args.savesamples, outdir=args.outdir)

if __name__ == '__main__':
    main()
