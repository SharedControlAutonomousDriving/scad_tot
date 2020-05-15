import os
import numpy as np
from tot_net import TOTNet
from utils import create_logger, count_decimal_places, ms_since_1970
from multiprocessing.pool import ThreadPool
from tensorflow.keras.models import load_model
from maraboupy import Marabou, MarabouUtils, MarabouCore

logger = create_logger('sensitivity')

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

def save_sensitivity_results_to_csv(results, outdir='../artifacts/sensitivity', outid=None):
    '''
    saves sensitivity summary and detailed results in csv format.

    @param results (dict): sensitivity results dictionary ({x0: (summary_tuple, details_list), ...})
    @param outdir (string): output directory
    @param outid (string): output file id
    '''
    summary_lines = ''.join(['x','dneg','dpos'])
    details_lines = ''.join(['x','s','dneg','dpos'])
    for x_id, result in results.items():
        x = int(x_id[1:])
        summary, details = result
        summary_lines.append(','.join([x, summary[0], summary[1]]))
        details_lines.extend([','.join([x, s, detail[0], detail[1]]) for s,detail in enumerate(details)])
    summary_file = os.path.join(outdir, f'summary_{outid}.csv' if outid else 'summary.csv')
    details_file = os.path.join(outdir, f'details_{outid}.csv' if outid else 'details.csv')
    if not os.path.exists(outdir):
        os.mkdirs(outdir)
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
        logger(f'wrote summary to {summary_file}')
    with open(details_file, 'w') as f:
        f.writelines(details_lines)
        logger(f'wrote detils to {details_file}')

def find_feature_sensitivity_boundaries(model_path, x, samples, d_min=0.001, d_max=100, multithread=False, verbose=0):
    '''
    finds +/- sensitivity boundaries of a feature on a given set of input samples (note predictions must be correct)
    
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
        lbound = d_min
        ubound = d_max
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
                if verbose > 0: logger.warning(f'no {dlabel(sign)} found for x{x}_s{s}@p={prec}')
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
            nthread = pool.apply_async(find_distance, (i, -1))
            pthread = pool.apply_async(find_distance, (i, 1))
            results.append((-1*nthread.get(), pthread.get()))
    else:
        results = [(-1*find_distance(i,-1), find_distance(i,+1)) for i in range(len(samples))]
    
    negd = max([d for d in [r for r in zip(*results)][0] if d is not 0] or [0])
    posd = min([d for d in [r for r in zip(*results)][1] if d is not 0] or [0])
    if verbose > 0: logger.info(f'x{x}: ({negd}, {posd}) ({ms_since_1970() - start}ms)')

    return ((negd, posd), results)

def find_sensitivity_boundaries(model_path, samples, d_min=0.001, d_max=100, multithread=False, verbose=0, save_results=False):
    '''
    finds sensitivity for all features relative to the provided samples

    @param model_path (string): h5 or pb model path
    @param samples (list): list of samples
    @d_min (float): min distance to consider
    @d_max (float): max distance to consider
    @multithread (bool): perform + and - in parallel (default false)
    @verbose (int): extra logging (0, 1, 2) (default 0)
    @return (dict): {x0:((negdist,posdist), [(s0_ndist,s0_pdist)...(sN_ndist,sN_pdist)]),...}
    '''
    n_features = len(samples[0][0])
    results = {}
    for x in range(n_features):
        results[f'x{x}'] = find_feature_sensitivity_boundaries(model_path, x, samples, d_min=d_min, d_max=d_max, multithread=multithread, verbose=verbose)
    if save_results: save_sensitivity_results_to_csv(results)
    return results
