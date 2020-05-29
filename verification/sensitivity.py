#!./venv/bin/python3

import os
import numpy as np
from argparse import ArgumentParser
from utils import create_logger, count_decimal_places, ms_since_1970, TOTUtils
from tot_net import TOTNet

default_outdir = './logs/sensitivity'
default_emin = 0.0001
default_emax = 100.0
default_timeout = 0
logger = create_logger('sensitivity', logdir=default_outdir)

def save_sensitivity_results_to_csv(results, samples, outdir):
    '''
    saves sensitivity summary and detailed results in csv format.

    @param results (dict): sensitivity results dictionary ({x0: (summary_tuple, details_list), ...})
    @param outdir (string): output directory
    '''
    summary_lines = ['x,leps,ueps\n']
    details_lines = ['x,s,leps,ueps,expy,predy\n']
    for x_id, result in results.items():
        x = int(x_id[1:])
        summary, details = result
        summary_lines.append(','.join([str(x), str(summary[0]), str(summary[1])])+'\n')
        details_lines.extend([','.join([str(x), str(s), str(detail[0]), str(detail[1]), str(samples[s][1].index(max(samples[s][1])))])+'\n' for s,detail in enumerate(details)])
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

def find_counterexample(net, sample, x, epsilon, asym_side='', timeout=default_timeout, verbose=0):
    inputs, outputs = sample
    # create upper and lower bounds (in asym mode, zero out the other side's epsilon)
    l_epsilon = 0 if asym_side and asym_side == 'u' else epsilon
    u_epsilon = 0 if asym_side and asym_side == 'l' else epsilon
    lbs = inputs[0:x] + [inputs[x]-l_epsilon] + inputs[x+1:]
    ubs = inputs[0:x] + [inputs[x]+u_epsilon] + inputs[x+1:]
    # find y index of prediction
    y_idx = outputs.index(max(outputs))
    return net.find_counterexample(lbs, ubs, y_idx, timeout=timeout)

def find_epsilon_bounds(net, sample, x, e_min, e_max, e_prec, asym_side='', timeout=default_timeout, verbose=0):
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
            counterexample = find_counterexample(net, sample, x, e, asym_side=asym_side, timeout=timeout, verbose=verbose)
            if counterexample:
                # return epsilon lower & upper bounds if counterexample was found
                e_lb = epsilons[i-1] if i > 0 else round(e-lb, dp+1)
                e_ub = e
                return (e_lb, e_ub)
    return (e_min, e_max)

def find_epsilon(net, sample, x, e_min, e_max, e_prec, asym_side='', timeout=default_timeout, verbose=0):
    '''
    finds precise epsilon value within specified bounds
    '''
    dplaces = count_decimal_places(e_prec)
    epsilons = [round(e, dplaces) for e in np.arange(e_min, e_max, e_prec)]
    if verbose > 1: logger.info(f'searching {len(epsilons)} {asym_side+"_" if asym_side else ""}epsilons b/t {epsilons[0]} and {epsilons[-1]}')
    # binary search range of epsilons
    e = 0
    l = 0
    m = 0
    h = len(epsilons) - 1
    cex_found = False
    epsilon = None
    while l <= h:
        m = (h + l) // 2
        e = epsilons[m]
        counterexample = find_counterexample(net, sample, x, e, asym_side=asym_side, timeout=timeout, verbose=verbose)
        if counterexample:
            h = m - 1
            cex_found = True
        else:
            l = m + 1
            epsilon = e
    if cex_found:
        return epsilon if epsilon is not None else round(e-e_prec, dplaces)
    return 0

def test_sensitivity(nnet_path, samples, x_indexes=[], e_min=default_emin, e_max=default_emax, e_prec=None, asym=False, coarse_pass=True, save_results=False, save_samples=False, outdir=default_outdir, timeout=default_timeout, verbose=0):
    if not e_prec:
        dp_prec = count_decimal_places(e_min)+1
        e_prec = round(1/(10**dp_prec), dp_prec)
    assert(e_prec <= e_min)
    net = TOTNet(nnet_path)
    start = ms_since_1970()
    x_indexes = x_indexes if x_indexes else [x for x in range(net.get_num_inputs())]
    results = {}
    for x in x_indexes:
        xid, x_start = f'x{x}', ms_since_1970()
        epsilons = []
        for s,sample in enumerate(samples):
            sid = f's{s}'
            if asym:
                le_bounds, ue_bounds = (e_min, e_max), (e_min, e_max)
                if coarse_pass:
                    # find coarse bounds for lower and upper epsilon
                    step_start = ms_since_1970()
                    le_bounds = find_epsilon_bounds(net, sample, x, e_min, e_max, e_prec, asym_side='l', timeout=timeout, verbose=verbose)
                    if verbose > 1: logger.info(f'{xid}_{sid} lower epsilon coarse bounds: {le_bounds} ({ms_since_1970() - step_start}ms)')
                    step_start = ms_since_1970()
                    ue_bounds = find_epsilon_bounds(net, sample, x, e_min, e_max, e_prec, asym_side='u', timeout=timeout, verbose=verbose)
                    if verbose > 1: logger.info(f'{xid}_{sid} upper epsilon coarse bounds: {ue_bounds} ({ms_since_1970() - step_start}ms)')
                # find lower and upper epsilon within the coarse bounds
                step_start = ms_since_1970()
                le = find_epsilon(net, sample, x, le_bounds[0], le_bounds[1], e_prec, asym_side='l', timeout=timeout, verbose=verbose)
                if verbose > 0: logger.info(f'{xid}_{sid} lower epsilon: {le} ({ms_since_1970() - step_start}ms)')
                step_start = ms_since_1970()
                ue = find_epsilon(net, sample, x, ue_bounds[0], ue_bounds[1], e_prec, asym_side='u', timeout=timeout, verbose=verbose)
                if verbose > 0: logger.info(f'{xid}_{sid} upper epsilon: {ue} ({ms_since_1970() - step_start}ms)')
                epsilons.append((le, ue))
            else:
                e_bounds = (e_min, e_max)
                if coarse_pass:
                    # find coarse bounds for epsilon
                    step_start = ms_since_1970()
                    e_bounds = find_epsilon_bounds(net, sample, x, e_min, e_max, e_prec, timeout=timeout, verbose=verbose)
                    if verbose > 1: logger.info(f'{xid}_{sid} coarse epsilon bounds: {e_bounds} ({ms_since_1970() - step_start}ms)')
                # find epsilon within ebounds
                step_start = ms_since_1970()
                e = find_epsilon(net, sample, x, e_bounds[0], e_bounds[1], e_prec, timeout=timeout, verbose=verbose)
                if verbose > 0: logger.info(f'{xid}_{sid} epsilon: {e} ({ms_since_1970() - step_start}ms)')
                epsilons.append((e, e))
        
        leps = [le for le,_ in epsilons if le != 0]
        ueps = [ue for _,ue in epsilons if ue != 0]
        x_summary = (-min(leps if leps else [0]), min(ueps if ueps else [0]))
        if verbose > 0: logger.info(f'{xid} epsilon: {x_summary} ({ms_since_1970() - x_start}ms)')
        results[f'x{x}'] = (x_summary, epsilons)
    
    summary = {x:r[0] for x,r in results.items()}
    logger.info(('asymm ' if asym else '') + f'sensitivity: {summary} ({ms_since_1970() - start}ms)')
    if save_results: save_sensitivity_results_to_csv(results, samples, outdir)
    if save_samples: TOTUtils.save_samples_to_csv(samples, outdir)
    return results

def check_sensitivity(nnet_path, samples, results, asym=False, find_multiple=False, outdir=default_outdir, timeout=default_timeout, verbose=0):
    def check_input_epsilon(net, sample, x, le, ue, asym=False, find_multiple=False):
        inputs, outputs = sample
        y = outputs.index(max(outputs))
        lbs = inputs[0:x] + [inputs[x]+le] + inputs[x+1:]
        ubs = inputs[0:x] + [inputs[x]+ue] + inputs[x+1:]
        cexs = None
        if asym:
            cexs = (
                net.find_counterexample(lbs, inputs, y, find_multiple=find_multiple),
                net.find_counterexample(inputs, ubs, y, find_multiple=find_multiple)
                )
        else:
            cexs = net.find_counterexample(lbs, ubs, y, find_multiple=find_multiple)
        return cexs
    
    net = TOTNet(nnet_path)
    test_results = {}
    for xid,result in results.items():
        x = int(xid[1:])
        le, ue = result[0]
        test_results[xid] = []
        for sample in samples:
            counterexamples = check_input_epsilon(net, sample, x, le, ue, asym=asym, find_multiple=find_multiple)
            test_results[xid].append(counterexamples)
        n_cexs = len([c for c in test_results[xid] if c]) if not asym else len([c for c in test_results[xid] if c[0] or c[1]])
        if not n_cexs:
            logger.info(f'{xid} ok {le, ue}')
        else:
            logger.info(f'{xid} counterexamples found for {n_cexs} samples {le, ue}:\n{test_results[xid]}')
    return test_results

if __name__ == '__main__':
    '''
    Usage: python3 verification/sensitivity.py -n NNETPATH -d DATAPATH [-df FRAC -x 0 1 2 -emin EMIN -emax EMAX -eprec EPREC -a -t TIMEOUT -sr -ss -sl -o OUTDIR -v V]
    '''
    parser = ArgumentParser()
    parser.add_argument('-n', '--nnetpath', required=True)
    parser.add_argument('-d', '--datapath', required=True)
    parser.add_argument('-df', '--datafrac', type=float, default=1)
    parser.add_argument('-x', '--xindexes', type=int, nargs='+')
    parser.add_argument('-emin', '--emin', type=float, default=default_emin)
    parser.add_argument('-emax', '--emax', type=float, default=default_emax)
    parser.add_argument('-eprec', '--eprec', type=float, default=None)
    parser.add_argument('-a', '--asym', action='store_true')
    parser.add_argument('-nc', '--nocoarse', action='store_true')
    parser.add_argument('-t', '--timeout', type=int, default=default_timeout)
    parser.add_argument('-sr', '--saveresults', action='store_true')
    parser.add_argument('-ss', '--savesamples', action='store_true')
    parser.add_argument('-sl', '--savelogs', action='store_true')
    parser.add_argument('-o', '--outdir', default=default_outdir)
    parser.add_argument('-v', '--verbose', type=int, default=0)
    args = parser.parse_args()
    # configure logger
    for handler in logger.handlers[:]: logger.removeHandler(handler)    
    logger = create_logger('sensitivity', to_file=args.savelogs, logdir=args.outdir)
    # load % of samples, and filter out incorrect predictions
    samples = TOTUtils.filter_samples(TOTUtils.load_samples(args.datapath, frac=args.datafrac), args.nnetpath)
    x_count = len(samples[0][0]) if not args.xindexes else len(args.xindexes)
    logger.info(f'starting sensitivity test for {x_count} features on {len(samples)} samples')
    results = test_sensitivity(args.nnetpath, samples, x_indexes=args.xindexes, e_min=args.emin, e_max=args.emax, e_prec=args.eprec, asym=args.asym, coarse_pass=not args.nocoarse, timeout=args.timeout, save_results=args.saveresults, save_samples=args.savesamples, outdir=args.outdir, verbose=args.verbose)
    logger.info(f'{"asym " if args.asym else ""}sensitivity results: {results[0]}')
