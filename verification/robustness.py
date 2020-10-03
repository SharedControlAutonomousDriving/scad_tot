#!./venv/bin/python3

import os
import numpy as np
from argparse import ArgumentParser
from utils import create_logger, create_dirpath, count_decimal_places, ms_since_1970, TOTUtils
from tot_net import TOTNet
from scipy.spatial import distance

default_outdir = './logs/robustness'
default_emin = 0.0001
default_emax = 100.0
default_timeout = 0
logger = create_logger('robustness', logdir=default_outdir)

def save_local_robustness_results_to_csv(results, samples, outdir):
    '''
    saves local robustness summary and detailed results in csv format.
    '''
    n_inputs, n_outputs = len(samples[0][0]), len(samples[0][1])

    summary_lines = ['leps,ueps\n']
    details_lines = [
        's,leps,ueps,spred,' + 
        ','.join([f'cex_x{x}' for x in range(n_inputs)] + [f'cex_y{y}' for y in range(n_outputs)]) +
        '\n']
    summary, details = results
    summary_lines.append(','.join([str(summary[0]), str(summary[1])])+'\n')
    for s,detail in enumerate(details):
        leps, ueps, spred, cex = detail[0], detail[1], samples[s][1].index(max(samples[s][1])), detail[2]
        cex = cex[0] if cex else ([0 for i in range(n_inputs)], [0 for i in range(n_outputs)])
        details_lines.append(','.join([str(s), str(leps), str(ueps), str(spred)] + [str(x) for x in cex[0]] + [str(y) for y in cex[1]]) + '\n')
    create_dirpath(outdir)
    summary_file = os.path.join(outdir, 'local_summary.csv')
    details_file = os.path.join(outdir, 'local_details.csv')
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
        logger.info(f'wrote summary to {summary_file}')
    with open(details_file, 'w') as f:
        f.writelines(details_lines)
        logger.info(f'wrote detils to {details_file}')

def find_counterexample(net, sample, epsilon, asym_side='', target_y=None, allowed_misclassifications=[], timeout=default_timeout, verbose=0):
    '''
    finds counterexample where classification changed for a given epsilon. None if no counterexamples found.
    '''
    inputs, outputs = sample
    l_epsilon = 0 if asym_side and asym_side == 'u' else epsilon
    u_epsilon = 0 if asym_side and asym_side == 'l' else epsilon
    # asym mode adjusts lower or upper side individually
    lbs = [x-l_epsilon for x in inputs]
    ubs = [x+u_epsilon for x in inputs]
    y_idx = outputs.index(max(outputs)) if target_y is None else target_y
    return net.find_counterexample(lbs, ubs, y_idx, inverse=(target_y is not None), allowed_misclassifications=allowed_misclassifications, timeout=timeout)

def find_epsilon_bounds(net, sample, e_min, e_max, e_prec, asym_side='', target_y=None, timeout=default_timeout, verbose=0):
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
            counterexample = find_counterexample(net, sample, e, asym_side=asym_side, target_y=target_y, timeout=timeout, verbose=verbose)
            if counterexample:
                # return epsilon lower & upper bounds if counterexample was found
                e_lb = epsilons[i-1] if i > 0 else round(e-(lb/10), dp+2)
                e_ub = e
                return (e_lb, e_ub)
    return (e_min, e_max)

def find_epsilon(net, sample, e_min, e_max, e_prec, asym_side='', target_y=None, allowed_misclassifications=[], timeout=default_timeout, verbose=0):
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
    counterexample = None
    epsilon = None
    while l < h:
        m = (h + l) // 2
        e = epsilons[m]
        cex = find_counterexample(net, sample, e, asym_side=asym_side, target_y=target_y, timeout=timeout, verbose=verbose, allowed_misclassifications=allowed_misclassifications)
        if cex:
            h = m - 1
            counterexample = cex
        else:
            l = m + 1
            epsilon = e
    if counterexample:
        return (epsilon, counterexample) if epsilon is not None else (round(e-e_prec, dplaces), counterexample)
    return (0, (([], []), None))

def test_local_robustness(nnet_path, samples, e_min=0.00001, e_max=100, e_prec=None, asym=False, coarse_pass=True, save_results=False, save_samples=False, outdir=default_outdir, timeout=default_timeout, verbose=0):
    if not e_prec:
        dp_prec = count_decimal_places(e_min)+1
        e_prec = round(1/(10**dp_prec), dp_prec)
    assert(e_prec < e_min)
    net = TOTNet(nnet_path)
    start = ms_since_1970()
    epsilons = []
    for s,sample in enumerate(samples):
        sid = f's{s}'
        if asym:
            l_ce_lb, l_ce_ub = (e_min, e_max), (e_min, e_max)
            if coarse_pass:
                # find coarse bounds for lower epsilon
                step_start = ms_since_1970()
                l_ce_lb, l_ce_ub = find_epsilon_bounds(net, sample, e_min, e_max, e_prec, asym_side='l', timeout=timeout, verbose=verbose)
                if verbose > 1: logger.info(f'{sid} lower epsilon coarse bounds: {l_ce_lb, l_ce_ub} ({ms_since_1970() - step_start}ms)')
                # find coarse bounds for upper epsilon
                step_start = ms_since_1970()
                u_ce_lb, u_ce_ub = find_epsilon_bounds(net, sample, e_min, e_max, e_prec, asym_side='u', timeout=timeout, verbose=verbose)
                if verbose > 1: logger.info(f'{sid} upper epsilon coarse bounds: {u_ce_lb, u_ce_ub} ({ms_since_1970() - step_start}ms)')
            # find lower epsilon within coarse bounds
            step_start = ms_since_1970()
            le, counterexample = find_epsilon(net, sample, l_ce_lb, l_ce_ub, e_prec, asym_side='l', timeout=timeout, verbose=verbose)
            if verbose > 0: logger.info(f'{sid} lower epsilon: {le} ({ms_since_1970() - step_start}ms)')
            # find upper epsilon within coarse bounds
            step_start = ms_since_1970()
            ue, counterexample = find_epsilon(net, sample, u_ce_lb, u_ce_ub, e_prec, asym_side='u', timeout=timeout, verbose=verbose)
            if verbose > 0: logger.info(f'{sid} upper epsilon: {ue} ({ms_since_1970() - step_start}ms)')
            epsilons.append((le, ue, counterexample))
        else:
            ce_lb, ce_ub = e_min, e_max
            if coarse_pass:
                # find coarse bounds for epsilon
                step_start = ms_since_1970()
                ce_lb, ce_ub = find_epsilon_bounds(net, sample, e_min, e_max, e_prec, timeout=timeout, verbose=verbose)
                if verbose > 1: logger.info(f'{sid} coarse epsilon bounds: {ce_lb, ce_ub} ({ms_since_1970() - step_start}ms)')
            # find epsilon within coarse bounds
            step_start = ms_since_1970()
            epsilon, counterexample = find_epsilon(net, sample, ce_lb, ce_ub, e_prec, timeout=timeout, verbose=verbose)
            if verbose > 0: logger.info(f'{sid} epsilon: {epsilon} ({ms_since_1970() - step_start}ms)')
            # update running min epislon
            epsilons.append((epsilon, epsilon, counterexample))
    # save and return test results
    leps = [le for le,_,_ in epsilons if le != 0]
    ueps = [ue for _,ue,_ in epsilons if ue != 0]
    summary = (-min(leps if leps else [0]), min(ueps if ueps else [0]))
    results = (summary, epsilons)
    logger.info(('asymm ' if asym else '') + f'local robustness: {summary} ({ms_since_1970() - start}ms)')
    if save_results: save_local_robustness_results_to_csv(results, samples, outdir)
    if save_samples: TOTUtils.save_samples_to_csv(samples, outdir)
    return results

def check_local_robustness(nnet_path, samples, results, asym=False, outdir=default_outdir, timeout=default_timeout, verbose=0):
    def check_epsilons(net, sample, le, ue, asym=False):
        cexs = []
        if asym:
            cexs = (
                find_counterexample(net, sample, le, asym_side='l', verbose=verbose),
                find_counterexample(net, sample, ue, asym_side='u', verbose=verbose)
            )
        else:
            cexs = find_counterexample(net, sample, ue, verbose=verbose)
        return cexs
    
    net = TOTNet(nnet_path)
    check_results = {}
    le, ue = results[0]
    for s,sample in enumerate(samples):
        sid = f's{s}'
        cexs = check_epsilons(net, sample, le, ue, asym=asym)
        check_results[sid] = cexs
    
    n_cexs = len([c for c in check_results.values() if c]) if not asym else len([c for c in check_results.values() if c[0] or c[1]])
    if not n_cexs:
        if verbose > 0: logger.info(f'{sid} ok {le, ue}')
    else:
        if verbose > 0: logger.info(f'counterexamples found for {n_cexs} samples {le, ue}\n{check_results}')
    return check_results

def test_targeted_robustness(nnet_path, samples, y, target_y, s_epsilons=[], e_min=0.0001, e_max=100, e_prec=0.00001, asym=False, coarse_pass=True, timeout=0, verbose=0):
    if s_epsilons:
        assert(len(samples) == len(s_epsilons))
    else:
        s_epsilons = [e_min for _ in samples]
    target_samples = {s:(i,o) for s,(i,o) in enumerate(samples) if o.index(1) == y}
    epsilon_bounds = {s:(s_epsilons[s], e_max) for s in target_samples}
    net = TOTNet(nnet_path)
    results = {}
    for s,sample in target_samples.items():
        # inputs, outputs = sample
        e_lb, e_ub = epsilon_bounds[s]
        if coarse_pass:
            step_start = ms_since_1970()
            e_lb, e_ub = find_epsilon_bounds(net, sample, e_lb, e_ub, e_prec, target_y=target_y, timeout=timeout, verbose=verbose)
            if verbose > 0: logger.info(f's{s} y{y}:y{target_y} coarse epsilon bounds {e_lb, e_ub} ({ms_since_1970() - step_start}ms)')
        e, cex = find_epsilon(net, sample, e_lb, e_ub, e_prec, target_y=target_y, timeout=timeout, verbose=verbose)
        if verbose > 0: logger.info(f's{s} y{y}:y{target_y} epsilon {e} ({ms_since_1970() - step_start}ms)')
        results[s] = (e, cex)
    return results

def verify_region(net, region, n_categories, eprec, rpad=1, verbose=0, timeout=0):
    radius, centroid, n_features = region.radius, region.centroid, region.X.shape[1]
    emax =  ((radius + rpad) / n_features)
    sample = (centroid, [int(region.category==i) for i in range(n_categories)])
    allowed_misclassifications = region.allowed_misclassifications if hasattr(region, 'allowed_misclassifications') else []
    start = ms_since_1970()
    vepsilon, cex = find_epsilon(net, sample, eprec, emax, eprec, allowed_misclassifications=allowed_misclassifications, verbose=verbose, timeout=timeout)
    duration = ms_since_1970() - start
    vradius = distance.euclidean(centroid + vepsilon, centroid)
    return (vradius, vepsilon, cex, duration)

def verify_regions(nnet_path, regions, n_categories, nmin=100, eprec=0.0001, rpad=1, verbose=0, timeout=0):
    nregions, vregions = len(regions), []
    net = TOTNet(nnet_path)
    for i,r in enumerate(regions):
        vrad, veps, cex, duration = verify_region(net, r, n_categories, eprec, rpad=rpad, verbose=verbose, timeout=timeout)
        if verbose > 0: logger.info(f'region {i} of {nregions} verified with r={vrad}, e={veps} ({duration} ms)')
        density = r.n/vrad if vrad > 0 else r.n/eprec
        vr = dict(centroid=r.centroid, radius=vrad, epsilon=veps, n=r.n, density=density, category=r.category, oradius=r.radius, counterexample=cex, duration=duration)
        vregions.append(vr)
    return vregions

def save_verified_regions(vregions, outdir=default_outdir, n_categories=5):
    n_features, n_categories = vregions[0]['centroid'].shape[0], n_categories
    header = ','.join(
        [f'cx{i}' for i in range(n_features)] + 
        ['radius', 'epsilon', 'n', 'density', 'category', 'oradius', 'duration'] + 
        [f'cex_x{x}' for x in range(n_features)] +
        [f'cex_y{y}' for y in range(n_categories)]
        )
    rows = []
    for r in vregions:
        cex = r['counterexample']
        cex = cex[0] if cex else (['' for i in range(n_features)], ['' for i in range(n_categories)])
        rows.append(','.join(
            [str(x) for x in r['centroid']] + 
            [str(v) for v in (r['radius'], r['epsilon'], r['n'], r['density'], r['category'], r['oradius'], r['duration'])] + 
            [str(x) for x in cex[0]] +
            [str(y) for y in cex[1]]
            ))
    create_dirpath(outdir)
    outpath = os.path.join(outdir, 'vregions.csv')
    with open(outpath, 'w') as f:
        f.writelines('\n'.join([header] + rows))
        logger.info(f'wrote verified regions to {outpath}')

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
    parser.add_argument('-nc', '--nocoarse', action='store_true')
    parser.add_argument('-t', '--timeout', default=default_timeout)
    parser.add_argument('-ck', '--checkresults', action='store_true')
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
    results = test_local_robustness(args.nnetpath, samples, e_min=args.emin, e_max=args.emax, e_prec=args.eprec, asym=args.asym, coarse_pass=not args.nocoarse, timeout=args.timeout, save_results=args.saveresults, save_samples=args.savesamples, outdir=args.outdir, verbose=args.verbose)        
    if args.checkresults:
        logger.info(f'checking {"asym " if args.asym else ""} robustness results...')
        check_results = check_local_robustness(args.nnetpath, samples, results, asym=args.asym, outdir=args.outdir, timeout=args.timeout, verbose=args.verbose)
        # notok = [k for k,v in check_results.items() if [i for i in v if any(i)]] if args.asym else [k for k,v in check_results.items() if any(v)]
        notok = [s for s,v in check_results.items() if any(v)] if args.asym else [s for s,v in check_results.items() if v]
        logger.info(f'local robustness check {"ok" if not notok else f"found counterexamples for {notok}"}')
        if notok: logger.info(f'local robustness check results {check_results}')
