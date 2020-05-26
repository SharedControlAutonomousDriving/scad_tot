#!./venv/bin/python3

import os
import numpy as np
from utils import create_logger, count_decimal_places, ms_since_1970, TOTUtils
from tot_net import TOTNet
from argparse import ArgumentParser

default_outdir = './logs/robustness'
default_emin = 0.0001
default_emax = 100.0
default_timeout = 0
logger = create_logger('robustness', logdir=default_outdir)

def save_local_robustness_results_to_csv(results, samples, outdir):
    '''
    saves local robustness summary and detailed results in csv format.
    '''
    summary_lines = ['leps,ueps\n']
    details_lines = ['s,leps,ueps\n']
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

def find_misclassification(net, sample, epsilon, asym_side='', timeout=default_timeout, verbose=0):
    '''
    finds counterexample where classification changed for a given epsilon. None if no counterexamples found.
    '''
    inputs, outputs = sample
    l_epsilon = 0 if asym_side and asym_side == 'u' else epsilon
    u_epsilon = 0 if asym_side and asym_side == 'l' else epsilon
    # asym mode adjusts lower or upper side individually
    lbs = [x-l_epsilon for x in inputs]
    ubs = [x+u_epsilon for x in inputs]
    y_idx = outputs.index(max(outputs))
    return net.find_misclassification(lbs, ubs, y_idx, timeout=timeout)

def find_epsilon_bounds(net, sample, e_min, e_max, e_prec, asym_side='', timeout=default_timeout, verbose=0):
    '''
    finds coarse lower and upper bounds for epsilon
    '''
    # count num places in decimal and mantissa
    dplaces = count_decimal_places(e_prec)
    mplaces = len(str(int(e_max)))
    # iterate through decimal places in reverse (e.g. 0.001, 0.01, 0.1, 1.0, 10.0)
    for dp in range(dplaces, -mplaces, -1):
        lb = round(1/(10**(dp+1)), (dp+1))
        ub = round(1/(10**dp), dp)
        epsilons = [round(e, dp+1) for e in np.arange(lb, ub, lb)]
        if verbose > 1: logger.info(f'searching {len(epsilons)} coarse {asym_side+"_" if asym_side else ""}epsilons b/t {epsilons[0]} and {epsilons[-1]}')
        for i,e in enumerate(epsilons):
            counterexample = find_misclassification(net, sample, e, asym_side=asym_side, timeout=timeout, verbose=verbose)
            if counterexample:
                # return epsilon lower & upper bounds if counterexample was found
                e_lb = epsilons[i-1] if i > 0 else round(e-lb, dp+1)
                e_ub = e
                return (e_lb, e_ub)
    return (e_min, e_max)

def find_epsilon(net, sample, e_min, e_max, e_prec, asym_side='', timeout=default_timeout, verbose=0):
    '''
    finds epsilon value within specified bounds by binary search at precision
    '''
    dplaces = count_decimal_places(e_prec)
    epsilons = [round(e, dplaces+1) for e in np.arange(e_min, e_max, e_prec)]
    if verbose > 1: logger.info(f'searching {len(epsilons)} {asym_side+"_" if asym_side else ""}epsilons b/t {epsilons[0]} and {epsilons[-1]}')
    e = 0
    l = 0
    m = 0
    h = len(epsilons) - 1
    cex_found = False
    epsilon = None
    while l < h:
        m = (h + l) // 2
        e = epsilons[m]
        counterexample = find_misclassification(net, sample, e, asym_side=asym_side, timeout=timeout, verbose=verbose)
        if counterexample:
            h = m - 1
            cex_found = True
        else:
            l = m + 1
            epsilon = e
    if cex_found:
        return epsilon if epsilon is not None else round(e-e_prec, dplaces)
    return 0

def test_local_robustness(nnet_path, samples, e_min=0.00001, e_max=100, e_prec=None, asym=False, save_results=False, save_samples=False, outdir=default_outdir, timeout=default_timeout, verbose=0):
    if not e_prec:
        dp_prec = count_decimal_places(e_min)+1
        e_prec = round(1/(10**dp_prec), dp_prec)
    assert(e_prec < e_min)
    net = TOTNet(nnet_path)
    start = ms_since_1970()
    epsilons = []
    for s,sample in enumerate(samples):
        if asym:
            # find coarse bounds for lower epsilon
            l_ce_lb, l_ce_ub = find_epsilon_bounds(net, sample, e_min, e_max, e_prec, asym_side='l', timeout=timeout, verbose=verbose)
            if verbose > 1: logger.info(f's{s} lower epsilon coarse bounds: {l_ce_lb, l_ce_ub}')
            # find coarse bounds for upper epsilon
            u_ce_lb, u_ce_ub = find_epsilon_bounds(net, sample, e_min, e_max, e_prec, asym_side='u', timeout=timeout, verbose=verbose)
            if verbose > 1: logger.info(f's{s} upper epsilon coarse bounds: {u_ce_lb, u_ce_ub}')
            # find lower epsilon within coarse bounds
            le = find_epsilon(net, sample, l_ce_lb, l_ce_ub, e_prec, asym_side='l', timeout=timeout, verbose=verbose)
            if verbose > 0: logger.info(f's{s} lower epsilon: {le}')
            # find upper epsilon within coarse bounds
            ue = find_epsilon(net, sample, u_ce_lb, u_ce_ub, e_prec, asym_side='u', timeout=timeout, verbose=verbose)
            if verbose > 0: logger.info(f's{s} upper epsilon: {ue}')
            epsilons.append((le, ue))
        else:
            # find coarse bounds for epsilon
            ce_lb, ce_ub = find_epsilon_bounds(net, sample, e_min, e_max, e_prec, timeout=timeout, verbose=verbose)
            if verbose > 1: logger.info(f's{s} coarse epsilon bounds: {ce_lb, ce_ub}')
            # find epsilon within coarse bounds
            epsilon = find_epsilon(net, sample, ce_lb, ce_ub, e_prec, timeout=timeout, verbose=verbose)
            if verbose > 0: logger.info(f's{s} epsilon: {epsilon}')
            # update running min epislon
            epsilons.append((epsilon, epsilon))
    
    # save and return test results
    leps = [le for le,_ in epsilons if le != 0]
    ueps = [ue for _,ue in epsilons if ue != 0]
    summary = (-min(leps if leps else [0]), min(ueps if ueps else [0]))
    results = (summary, epsilons)
    logger.info(('asymm ' if asym else '') + f'local robustness: {summary}')
    if save_results: save_local_robustness_results_to_csv(results, samples, outdir)
    if save_samples: TOTUtils.save_samples_to_csv(samples, outdir)
    return results

if __name__ == '__main__':
    '''
    Usage: python3 verification/robustness.py -n NNETPATH -d DATAPATH [-df FRAC -emin EMIN -emax EMAX -eprec EPREC -a -t TIMEOUT -sr -ss -sl -o OUTDIR -v V]
    '''
    parser = ArgumentParser()
    parser.add_argument('-n', '--nnetpath', required=True)
    parser.add_argument('-d', '--datapath', required=True)
    parser.add_argument('-df', '--datafrac', type=float, default=1)
    parser.add_argument('-emin', '--emin', type=float, default=default_emin)
    parser.add_argument('-emax', '--emax', type=float, default=default_emax)
    parser.add_argument('-eprec', '--eprec', type=float)
    parser.add_argument('-a', '--asym', action='store_true')
    parser.add_argument('-t', '--timeout', default=default_timeout)
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
    samples = TOTUtils.filter_samples(TOTUtils.load_samples(args.datapath, frac=args.datafrac), args.nnetpath)
    logger.info(f'starting local robustness test on {len(samples)} samples')
    results = test_local_robustness(args.nnetpath, samples, e_min=args.emin, e_max=args.emax, e_prec=args.eprec, asym=args.asym, timeout=args.timeout, save_results=args.saveresults, save_samples=args.savesamples, outdir=args.outdir, verbose=args.verbose)
    logger.info(f'local robustness results:', results[0])
