import numpy as np
from tot_net import TOTNet
from tensorflow.keras.models import load_model
import os
import numpy as np
from utils import create_logger, count_decimal_places, ms_since_1970
from multiprocessing.pool import ThreadPool
from tensorflow.keras.models import load_model
from maraboupy import Marabou

logger = create_logger('robustness')

def save_local_robustness_results_to_csv(results, outdir='../artifacts/robustness', outid=None):
    '''
    saves local robustness summary and detailed results in csv format.

    @param results (dict): robustness results dictionary ({x0: (summary_tuple, details_list), ...})
    @param outdir (string): output directory
    @param outid (string): output file id
    '''
    summary_lines = ['dneg,dpos']
    details_lines = ['s,dneg,dpos']
    summary, details = results
    summary_lines.append(','.join([str(summary[0]), str(summary[1])]))
    details_lines.extend([','.join([str(s), str(detail[0]), str(detail[1])]) for s,detail in enumerate(details)])
    summary_file = os.path.join(outdir, f'local_summary_{outid}.csv' if outid else 'local_summary.csv')
    details_file = os.path.join(outdir, f'local_details_{outid}.csv' if outid else 'local_details.csv')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(summary_file, 'w') as f:
        f.writelines(summary_lines)
        logger.info(f'wrote summary to {summary_file}')
    with open(details_file, 'w') as f:
        f.writelines(details_lines)
        logger.info(f'wrote detils to {details_file}')

def find_local_robustness_boundaries(model_path, samples, d_min=0.001, d_max=100, multithread=False, verbose=0, save_results=False):
    '''
    finds local robustness of network based on input samples.

    @param model_path (string): h5 or pb model path
    @param samples (list): list of samples
    @d_min (float): min distance to consider
    @d_max (float): max distance to consider
    @multithread (bool): perform + and - in parallel (default false)
    @verbose (int): extra logging (0, 1, 2) (default 0)
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
            adj_s = [[x+(d*sign) for x in inputs] for d in distances]
            predictions = model.predict(adj_s) if len(adj_s) > 0 else []
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
            nthread = pool.apply_async(find_distance, (s, -1))
            pthread = pool.apply_async(find_distance, (s, 1))
            details.append((-1*nthread.get(), pthread.get()))
    else:
        details = [(-1*find_distance(i,-1), find_distance(i,+1)) for i in range(len(samples))]
    
    negd = max([d for d in [r for r in zip(*details)][0] if d is not 0] or [0])
    posd = min([d for d in [r for r in zip(*details)][1] if d is not 0] or [0])
    # return tuple containing summary tuple & details list
    results = ((negd, posd), details)
    if save_results: save_local_robustness_results_to_csv(results)
    return results


