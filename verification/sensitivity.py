import numpy as np
from tot_net import TOTNet
from utils import create_logger, count_decimal_places, ms_since_1970
from multiprocessing.pool import ThreadPool
from tensorflow.keras.models import load_model
from maraboupy import Marabou, MarabouUtils, MarabouCore

logger = create_logger('sensitivity')

def evaluate_sample(nnet_path, input_sample, output_sample):
    expected_cat = output_sample.index(max(output_sample))
    other_cats = [i for i in range(len(output_sample)) if i != expected_cat]
    net = Marabou.read_nnet(nnet_path)
    for x,v in enumerate(input_sample):
        net.setLowerBound(net.inputVars[0][x], v)
        net.setUpperBound(net.inputVars[0][x], v)
    pred = net.evaluate([input_sample])
    pred_cat = list(pred[0]).index(max(pred[0]))
    result = 'UNSAT' if pred_cat == expected_cat else 'SAT'
    return (result, pred)

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

def find_sensitivity_boundaries(model_path, samples, d_min=0.001, d_max=100, multithread=False, verbose=0):
    '''
    finds sensitivity for all features relative to the provided samples

    @param model_path (string): h5 or pb model path
    @param samples (list): list of samples
    @d_min (float): min distance to consider
    @d_max (float): max distance to consider
    @multithread (bool): perform + and - in parallel (default false)
    @verbose (int): extra logging (0, 1, 2) (default 0)
    '''
    n_features = len(samples[0][0])
    results = {}
    for x in range(n_features):
        result = find_feature_sensitivity_boundaries(model_path, x, samples, d_min=d_min, d_max=d_max, multithread=multithread, verbose=verbose)
        results[f'x{x}'] = result
    return results
    