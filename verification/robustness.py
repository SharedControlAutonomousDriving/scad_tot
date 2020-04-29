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

# TODO: Fix RecursionError stack overflow (iterative instead of recursion)
def find_feature_sensitivity_recursive_backtrack(nnet_path, x, samples, d_min=0.01, d_max=50.00, d_step=0.01):
    def find_shortest_distance(sample, distances, curr_i=0, prev_i=0):
        input_sample, output_sample = sample
        num_dist = len(distances)
        d = distances[curr_i]
        s = input_sample[0:x] + [input_sample[x]+d] + input_sample[x+1:]
        counterexamples = evaluate_sample(nnet_path, s, output_sample)
        result = 'UNSAT' if len(counterexamples) == 0 else 'SAT'
        jump_size = 2

        # UNSAT while moving forward. move forward i*jump_size steps.
        if result == 'UNSAT' and curr_i - prev_i >= 0:
            next_i = max(1, curr_i) + (curr_i - prev_i) * jump_size
            next_i = min(next_i, num_dist-1)
            return find_shortest_distance(sample, distances, next_i, curr_i)
        # UNSAT while moving backward. move forward to midpt b/t prev and curr.
        elif result == 'UNSAT' and prev_i - curr_i > 0:
            next_i = curr_i + (prev_i - curr_i) // 2
            return find_shortest_distance(sample, distances, next_i, curr_i)
        # SAT while moving backward more than 1 step. move backward.
        elif result == 'SAT' and prev_i - curr_i > 1:
            next_i = curr_i - (prev_i - curr_i) // 2
            return find_shortest_distance(sample, distances, next_i, curr_i)
        # SAT while moving forward more than 1 step. move forward.
        elif result == 'SAT' and curr_i - prev_i > 1:
            next_i = prev_i + (curr_i - prev_i) // 2
            return find_shortest_distance(sample, distances, next_i, curr_i)
        # SAT while moving backward by 1 step. done.
        else:
            return (d, counterexamples)

    distances = np.round(np.arange(d_min, d_max, d_step), count_decimal_places(d_step))
    results = []
    for sample in samples:
        neg_d, _ = find_shortest_distance(sample, distances * -1)
        pos_d, _ = find_shortest_distance(sample, distances)
        results.append((neg_d, pos_d))
        # prune distances between 0 and the max distance
        d_stop = np.where(distances==max(neg_d*-1, pos_d))[0][0]
        distances = distances[0:d_stop+1]
    return (max(list(zip(*results))[0]), min(list(zip(*results))[1]))

def find_feature_sensitivity_bruteforce(nnet_path, x, samples, d_min=0.01, d_max=100.00, d_step=0.01):
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
        # prune distances between 0 and the max distance
        d_stop = np.where(distances==max(l_dist, r_dist))[0][0]
        distances = distances[0:d_stop+1]
    return (-1*l_dist, r_dist)

def test_network_sensitivity(nnet_path, n_features, samples):
    results = {}
    for x in range(n_features):
        results[f'x{x}'] = find_feature_sensitivity_bruteforce(nnet_path, x, samples)
    return results
