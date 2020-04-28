import numpy as np
from tot_net import TOTNet
from bisect import bisect_left
from utils import count_decimal_places

def evaluate_sample(nnet_path, input_sample, output_sample):
    category_index = output_sample.index(max(output_sample))
    other_categories = [i for i in range(len(output_sample)) if i != category_index]
    counterexamples = []
    for c in other_categories:
        net = TOTNet(nnet_path)
        net.set_lower_bounds(input_sample)
        net.set_upper_bounds(input_sample)
        net.set_expected_category(c)
        vals, _ = net.solve()
        if len(vals) > 0:
            counterexamples.append(vals)
    return counterexamples

def find_feature_sensitivity(nnet_path, x, samples, d_min=0.01, d_max=100.00, d_step=0.01):
    l_dist = 0
    r_dist = 0
    distances = np.round(np.arange(d_min, d_max, d_step), count_decimal_places(d_step))
    for i,(input_sample, output_sample) in enumerate(samples):
        for d in distances:
            if d >= l_dist:
                l_sample = input_sample[0:x] + [input_sample[x]-d] + input_sample[x+1:]
                l_result = evaluate_sample(nnet_path, l_sample, output_sample)
                if len(l_result) > 0:
                    l_dist = d
                    break
        for d in distances:
            if d >= r_dist:
                r_sample = input_sample[0:x] + [input_sample[x]+d] + input_sample[x+1:]
                r_result = evaluate_sample(nnet_path, r_sample, output_sample)
                if len(r_result) > 0:
                    r_dist = d
                    break
    return (l_dist, r_dist)

def find_feature_sensitivity2(nnet_path, x, samples, d_min=0.01, d_max=50.00, d_step=0.01):
    def find_min_distance(sample, distances, curr_i=0, prev_i=0):
        input_sample, output_sample = sample
        d = distances[curr_i]
        s = input_sample[0:x] + [input_sample[x]+d] + input_sample[x+1:]
        counterexamples = evaluate_sample(nnet_path, s, output_sample)
        result = 'UNSAT' if len(counterexamples) == 0 else 'SAT'

        # # SAT while moving backward by 1 step. done.
        # if result == 'SAT' and prev_i - curr_i == 1:
        #     print('A')
        #     return (d, result, counterexamples)
        
        # UNSAT while moving forward. move forward i*2 steps.
        if result == 'UNSAT' and curr_i - prev_i >= 0:
            next_i = max(1, curr_i) + (curr_i - prev_i) * 2
            return find_min_distance(sample, distances, next_i, curr_i)
        
        # UNSAT while moving backward. move forward to midpt b/t prev and curr.
        elif result == 'UNSAT' and prev_i - curr_i > 0:
            next_i = curr_i + (prev_i - curr_i) // 2
            return find_min_distance(sample, distances, next_i, curr_i)
        
        # SAT while moving backward more than 1 step. move backward.
        elif result == 'SAT' and prev_i - curr_i > 1:
            next_i = curr_i - (prev_i - curr_i) // 2
            return find_min_distance(sample, distances, next_i, curr_i)
        
        # SAT while moving forward more than 1 step. move forward.
        elif result == 'SAT' and curr_i - prev_i > 1:
            next_i = prev_i + (curr_i - prev_i) // 2
            return find_min_distance(sample, distances, next_i, curr_i)
        
        # SAT while moving backward by 1 step. done.
        elif result == 'SAT' and prev_i - curr_i == 1:
            return (d, result, counterexamples)

    distances = np.round(np.arange(d_min, d_max, d_step), count_decimal_places(d_step))
    pos_result = find_min_distance(samples[0], distances)
    neg_result = find_min_distance(samples[0], distances * -1)
    return (pos_result, neg_result)

def test_network_sensitivity(nnet_path, n_features, samples):
    results = {}
    for x in range(n_features):
        results[f'x{x}'] = find_feature_sensitivity(nnet_path, x, samples)
    return results