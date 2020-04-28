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

        print('C', curr_i, 'P', prev_i, 'D', d, 'R', result)
        
        # if result == 'SAT' and curr_d - prev_d == 1 or ((curr_d == prev_d) and curr_d != 0):
        # SAT while moving backward by 1 step.  DONE.
        if result == 'SAT' and prev_i - curr_i == 1:
            print('DONE.')
            return (result, d)
        
        # SAT while moving backward. move backwards.
        elif result == 'SAT' and prev_i - curr_i >= 0:
            next_i = curr_i - (prev_i - curr_i)//2
            print('SAT backward. next:', next_i)
            return find_min_distance(sample, distances, next_i, curr_i)
        
        # SAT while moving forward. move forward.
        elif result == 'SAT' and curr_i - prev_i >= 0:
            next_i = prev_i + (curr_i - prev_i)//2
            print('SAT forward. next:', next_i)
            return find_min_distance(sample, distances, next_i, curr_i)

        # UNSAT while moving forward. move forward.
        elif result == 'UNSAT' and curr_i - prev_i >= 0:
            next_i = (curr_i if curr_i > 0 else 1) + (curr_i - prev_i) * 2
            print('UNSAT forward. next:', next_i)
            return find_min_distance(sample, distances, next_i, curr_i)
        
        # UNSAT while moving backward. move forward.
        elif result == 'UNSAT' and prev_i - curr_i >= 0:
            # next_d = (curr_d if curr_d > 0 else 1) + (curr_d - prev_d) * 2
            next_i = curr_i + (prev_i - curr_i)//2
            print('UNSAT backward. next:', next_i)
            return find_min_distance(sample, distances, next_i, curr_i)

    distances = np.round(np.arange(d_min, d_max, d_step), count_decimal_places(d_step))
    # return b_search(distances, samples[0], 0, len(distances), 0)
    return find_min_distance(samples[0], distances)

def test_network_sensitivity(nnet_path, n_features, samples):
    results = {}
    for x in range(n_features):
        results[f'x{x}'] = find_feature_sensitivity(nnet_path, x, samples)
    return results