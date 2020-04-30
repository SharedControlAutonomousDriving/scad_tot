import numpy as np
from tot_net import TOTNet
from bisect import bisect_left
from utils import count_decimal_places
from maraboupy import Marabou, MarabouUtils, MarabouCore

# def evaluate_sample(nnet_path, input_sample, output_sample):
#     category_index = output_sample.index(max(output_sample))
#     other_categories = [i for i in range(len(output_sample)) if i != category_index]
#     counterexamples = []
#     for c in other_categories:
#         net = TOTNet(nnet_path)
#         net.set_lower_bounds(input_sample)
#         net.set_upper_bounds(input_sample)
#         net.set_expected_category(c)
#         vals, _ = net.solve()
#         if len(vals) > 0:
#             counterexamples.append(vals)
#     return counterexamples

# net.set_expected_category(c)
# other_cats_y = [y for y in range(other_cats_y) if y is not other_cats_y]
# for other_y in other_cats_y:
#     eq = MarabouCore.Equation(MarabouCore.Equation.LE)
#     eq.addAddend(1, net.outputVars[0][other_y])
#     eq.addAddend(-1, net.outputVars[0][c])
#     eq.setScalar(0)
#     net.addEquation(eq)
#     # vals, _ = net.solve()

#     if len(vals) > 0:
#         results.append(vals)

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

# NOTE: May throw a RecursionError or crash due to exceeding memory limits
def find_feature_sensitivity_recursive_binary_search(nnet_path, x, samples, d_min=0.01, d_max=50.00, d_step=0.01):
    def find_shortest_distance(sample, distances, curr_i=0, prev_i=0):
        input_sample, output_sample = sample
        num_dist = len(distances)
        d = distances[curr_i]
        s = input_sample[0:x] + [input_sample[x]+d] + input_sample[x+1:]
        result,prediction = evaluate_sample(nnet_path, s, output_sample)
        # result = 'UNSAT' if len(counterexamples) == 0 else 'SAT'
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
            return (d, prediction)

    distances = np.round(np.arange(d_min, d_max, d_step), count_decimal_places(d_step))
    results = []
    for sample in samples:
        neg_d, _ = find_shortest_distance(sample, distances * -1)
        pos_d, _ = find_shortest_distance(sample, distances)
        results.append((neg_d, pos_d))
    return (max(list(zip(*results))[0]), min(list(zip(*results))[1]))

def find_feature_sensitivity_distances(nnet_path, x, samples, d_min=0.01, d_max=50.00, d_step=0.01):
    def find_distance(sample, sign):
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
                return d
            prev_d = d
            d = next_d

    results = [(-1*find_distance(s, -1), find_distance(s, +1)) for s in samples]
    return (max(list(zip(*results))[0]), min(list(zip(*results))[1]))


def find_feature_sensitivity_binary_search(nnet_path, x, samples, d_min=0.01, d_max=50.00, d_step=0.01):
    def find_shortest_distance(sample, distances):
        curr_i, prev_i = 0, 0
        inputs, outputs = sample
        num_dist = len(distances)
        checked = np.zeros(num_dist)
        jump_size = 2
        sat_found = False
        while not all(checked):
            checked[curr_i] = 1
            d = distances[curr_i]
            s = inputs[0:x] + [inputs[x]+d] + inputs[x+1:]
            result,prediction = evaluate_sample(nnet_path, s, outputs)
            if result == 'SAT': sat_found = True
            print('C:', curr_i, 'P:', prev_i, 'D:', d, 'R:', result)

            # if result == 'UNSAT' and curr_i - prev_i >= 0:
            if result == 'UNSAT' and curr_i == 0 and prev_i == 0:
                print('0')
                next_i = 1
            # UNSAT while moving forward. move forward i*jump_size steps.
            elif result == 'UNSAT' and curr_i - prev_i >= 0:
                print('A')
                if not sat_found:
                    next_i = curr_i + (curr_i - prev_i) * 2
                else:
                    next_i = curr_i + (curr_i - prev_i) // 2
                # next_i = max(1, curr_i) + (curr_i - prev_i) * jump_size
                next_i = min(next_i, num_dist-1)
            # UNSAT while moving backward. move forward to midpt b/t prev and curr.
            elif result == 'UNSAT' and prev_i - curr_i > 0:
                print('B')
                next_i = curr_i + (prev_i - curr_i) // 2
            # SAT while moving backward more than 1 step. move backward.
            elif result == 'SAT' and prev_i - curr_i > 1:
                print('C')
                next_i = curr_i - (prev_i - curr_i) // 2
            # SAT while moving forward more than 1 step. move forward.
            elif result == 'SAT' and curr_i - prev_i > 1:
                next_i = prev_i + (curr_i - prev_i) // 2
            # SAT while moving backward by 1 step. done.
            else:
                return (d, prediction)
            prev_i = curr_i
            curr_i = next_i

    distances = np.round(np.arange(d_min, d_max, d_step), count_decimal_places(d_step))
    results = []
    for sample in samples:
        neg_d, _ = find_shortest_distance(sample, distances * -1)
        pos_d, _ = find_shortest_distance(sample, distances)
        results.append((neg_d, pos_d))
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
                l_result,_ = evaluate_sample(nnet_path, l_sample, output_sample)
                # if len(l_result) > 0:
                if l_result == 'SAT':
                    l_dist = d
                    break
        for d in distances:
            if d >= r_dist:
                r_sample = input_sample[0:x] + [input_sample[x]+d] + input_sample[x+1:]
                # r_result = evaluate_sample(nnet_path, r_sample, output_sample)
                r_result,_ = evaluate_sample(nnet_path, r_sample, output_sample)
                # if len(r_result) > 0:
                if r_result == 'SAT':
                    r_dist = d
                    break
    return (-1*l_dist, r_dist)

def test_network_sensitivity(nnet_path, n_features, samples):
    results = {}
    for x in range(n_features):
        print(f'starting x{x}')
        result = find_feature_sensitivity_bruteforce(nnet_path, x, samples)
        print(f'x{x}: ', result)
        results[f'x{x}'] = result
    return results
