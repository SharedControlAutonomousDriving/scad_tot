import numpy as np
import time
from tot_net import TOTNet
from bisect import bisect_left
from utils import count_decimal_places
from maraboupy import Marabou, MarabouUtils, MarabouCore
from tensorflow.keras.models import load_model

ms_since_1970 = lambda: int(round(time.time() * 1000))

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

def find_feature_sensitivity_boundaries_binarysearch(nnet_path, x, samples, d_min=0.01, d_max=100.00, verbose=False):
    def find_distance(sample, s_num, sign):
        start = ms_since_1970()
        prev_d = 0
        d = d_min
        inputs, outputs = sample
        while(d < d_max):
            s = inputs[0:x] + [inputs[x]+d*sign] + inputs[x+1:]
            result,prediction = evaluate_sample(nnet_path, s, outputs)
            # unsat on forward movement. move forward.
            if result == 'UNSAT' and d > prev_d:
                next_d = d + (d - prev_d) * 2
            # unsat on backward movement. move forward halfway b/t d and prev.
            elif result == 'UNSAT' and d < prev_d:
                next_d = d + (prev_d - d) / 2
            # sat on forward movement. move backward halfway b/t prev and d
            elif result == 'SAT' and d > prev_d:
                next_d = d - (d - prev_d) / 2
            # sat on backward movement. move backward halfway b/t d and prev.
            elif result == 'SAT' and d < prev_d:
                next_d = d - (prev_d - d) / 2
            else:
                if verbose: print(f's{s_num}_x{x} {"+" if sign == 1 else "-1"}dist: {sign * d} ({ms_since_1970()-start}ms)')
                return d
            prev_d = d
            d = next_d
        return 0
    results = [(-1*find_distance(s, i, -1), find_distance(s, i, +1)) for i,s in enumerate(samples)]
    ld = max([d for d in [r for r in zip(*results)][0] if d is not 0] or [0])
    rd = min([d for d in [r for r in zip(*results)][1] if d is not 0] or [0])
    return ((ld, rd), results)

def find_feature_sensitivity_boundaries_bruteforce(nnet_path, x, samples, d_min=0.001, d_max=100.00, d_step=0.0001, verbose=False):
    def find_distance(sample, s_num, sign):
        start = ms_since_1970()
        inputs, outputs = sample
        d = d_min
        while(d < d_max):
            s = inputs[0:x] + [inputs[x]+(d*sign)] + inputs[x+1:]
            result,_ = evaluate_sample(nnet_path, s, outputs)
            if result == 'SAT':
                if verbose: print(f's{s_num}_x{x} {"+" if sign == 1 else "-1"}dist: {sign * d} ({ms_since_1970()-start}ms)')
                return d
            d += d_step
        return 0
    results = [(-1*find_distance(s, i, -1), find_distance(s, i, +1)) for i,s in enumerate(samples)]
    ld = max([d for d in [r for r in zip(*results)][0] if d is not 0] or [0])
    rd = min([d for d in [r for r in zip(*results)][1] if d is not 0] or [0])
    return ((ld, rd), results)

def find_feature_sensitivity_boundaries_bruteforce_optimized(h5_path, x, samples, d_min=0.001, d_max=100.00, d_step=0.0001, verbose=False):
    def find_distance(sample, s_num, sign):
        start = ms_since_1970()
        inputs, outputs = sample
        exp_cat = outputs.index(max(outputs))
        distances = np.arange(d_min, d_max, d_step)
        adj_s = [inputs[0:x]+[inputs[x]+(d*sign)]+inputs[x+1:] for d in distances]
        predictions = model.predict(adj_s)
        for i,pred in enumerate(predictions):
            cat = list(pred).index(max(pred))
            if cat != exp_cat:
                if verbose: print(f's{s_num}_x{x} {"+" if sign == 1 else "-1"}dist: {sign * distances[i]} ({ms_since_1970()-start}ms)')
                return distances[i]
        return 0
    model = load_model(h5_path)
    results = [(-1*find_distance(s, i, -1), find_distance(s, i, +1)) for i,s in enumerate(samples)]
    ld = max([d for d in [r for r in zip(*results)][0] if d is not 0] or [0])
    rd = min([d for d in [r for r in zip(*results)][1] if d is not 0] or [0])
    return ((ld, rd), results)

def find_sensitivity_boundaries(nnet_path, samples, d_min=0.001, d_max=100, d_step=0.0001, bruteforce=True, use_marabou=True, verbose=False):
    n_features = len(samples[0][0])
    results = {}
    for x in range(n_features):
        if bruteforce:
            if use_marabou:
                result = find_feature_sensitivity_boundaries_bruteforce(nnet_path, x, samples, d_min=d_min, d_max=d_max, d_step=d_step, verbose=verbose)
            else:
                result = find_feature_sensitivity_boundaries_bruteforce_optimized(nnet_path, x, samples, d_min=d_min, d_max=d_max, d_step=d_step, verbose=verbose)
        else:
            result = find_feature_sensitivity_boundaries_binarysearch(nnet_path, x, samples, d_min=d_min, d_max=d_max, verbose=verbose)
        results[f'x{x}'] = result
        print(f'x{x}: ', result[0])
    return results


def find_feature_sensitivity_boundaries_x(model_path, x, samples, d_min=0.0001, d_max=100, verbose=False):
    def find_distance(s_idx, sign):
        start = ms_since_1970()
        inputs, outputs = samples[s_idx]
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
                    print(f'break @ {dist}, newbounds: ({lbound},{ubound})')
                    break
            # print(f'dp={dp}, dist={dist}')
        # print(f'{"-" if sign < 0 else "+"}dist: {dist}')
        return dist
    dplaces = count_decimal_places(d_min)
    model = load_model(model_path)
    results = [(-1*find_distance(i, -1), find_distance(i, +1)) for i in range(len(samples))]
    ld = max([d for d in [r for r in zip(*results)][0] if d is not 0] or [0])
    rd = min([d for d in [r for r in zip(*results)][1] if d is not 0] or [0])
    return ((ld, rd), results)


# def find_dist(target, d_min, d_max):
#     dplaces = count_decimal_places(d_min)
#     lb = d_min
#     ub = d_max
#     d = 0
#     for dp in range(dplaces+1):
#         prec = round(1/(10**dp), dp)
#         dists = np.arange(lb, ub, prec)
#         for dist in dists:
#             if dist > target:
#                 lb = dist - prec if dist - prec > 0 else d_min
#                 ub = dist
#                 d = dist
#                 break
#         print(f'Bounds: ({lb}, {ub})')
#     return d
    