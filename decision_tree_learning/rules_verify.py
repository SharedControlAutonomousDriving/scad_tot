import sys
# add the submodules to $PATH
# sys.path[0] is the current file's path
sys.path.append(sys.path[0] + '/..')

import os
import pickle
import pandas as pd
from verification.tot_net import TOTNetV1
import pickle
import numpy as np
import pandas as pd
from itertools import product
from maraboupy import Marabou, MarabouCore, MarabouUtils
from scriptify import scriptify

# 1. ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence > -1.704 : class = TOT_med_slow
# 2. ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH <= -2.126 : CLASS = TOT_med
# 3. ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH > -2.126 : CLASS = TOT_med_fast
# 4. ManualWheel <= 0.307 & FixationStart > -1.677 & FixationStart > 1.717 & PupilLeft > 0.424 & FixationStart > 1.924 : CLASS = TOT_slow
# 5. ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel <= 0.887 : CLASS: TOT_med_slow
# 6. ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel > 0.887 : CLASS: TOT_fast

rules_25 = [
    # ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence > -1.704 : class = TOT_med_slow
    {
        'name': 'Rule1',
        'ubs': {'ManualWheel': 0.307, 'FixationStart': -1.677, 'MPH': -1.34},
        'lbs':{'FixationSeq': -1.704},
        'out': 'med_slow'
    },
    # ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH <= -2.126 : CLASS = TOT_med
    {
        'name': 'Rule2',
        'ubs': {'ManualWheel': 0.307, 'FixationStart': -1.677, 'MPH': -2.126, 'FixationSeq': -1.704},
        'lbs':{},
        'out': 'med'
    },
    # ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH > -2.126 : CLASS = TOT_med_fast
    {
        'name': 'Rule3',
        'ubs': {'ManualWheel': 0.307, 'FixationStart': -1.677, 'MPH': -1.34, 'FixationSeq': -1.704},
        'lbs':{'MPH': -2.126},
        'out': 'med_fast'
    },
    # ManualWheel <= 0.307 & FixationStart > -1.677 & FixationStart > 1.717 & PupilLeft > 0.424 & FixationStart > 1.924 : CLASS = TOT_slow
    {
        'name': 'Rule4',
        'ubs': {'ManualWheel': 0.307},
        'lbs': {'PupilLeft': 0.424, 'FixationStart':1.924},
        'out': 'slow'
    },
    # ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel <= 0.887 : CLASS: TOT_med_slow
    {
        'name': 'Rule5',
        'ubs': {'ManualWheel': 0.461, 'FixationSeq': 0.62, 'AutoWheel': 0.887},
        'lbs': {'ManualWheel': 0.307, 'InterpolatedGazeY': 0.109, 'Distance3D': 0.283, 'FixationSeq': -0.043},
        'out': 'med_slow'
    },
    # ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel > 0.887 : CLASS: TOT_fast
    {
        'name': 'Rule6',
        'ubs': {'ManualWheel': 0.461, 'FixationSeq': 0.62},
        'lbs': {'ManualWheel': 0.307, 'InterpolatedGazeY': 0.109, 'Distance3D': 0.283, 'FixationSeq': -0.043, 'AutoWheel': 0.887},
        'out': 'fast'
    }
]

# 1. MPH <= 0.865 & FixationStart <= -1.663 & FixationStart <= -1.91 & PupilRight > -0.504 & ManualBreak <= 4.058 ----- class = TOT_slow
# 2. MPH <= 0.865 & FixationStart <= -1.663 & FixationStart <= -1.91 & PupilRight <= -0.504 & MPH <= -4.832 ----- class = TOT_slow
# 3. MPH <= 0.865 & FixationStart <= -1.663 & FixationStart <= -1.91 & PupilRight <= -0.504 & MPH > -4.832 ----- class = TOT_med_slow
# 4. MPH > 0.865 and FizationStart > -0.17 and GazeDirectionRightZ ≤ 0.503 and InterpolatedGazeY > -0.047 and MPH <= 1.191 and PupilRight > -0.681 ------ class = TOT_med_fast
rules_24 = [
    # MPH <= 0.865 & FixationStart <= -1.663 & FixationStart <= -1.91 & PupilRight > -0.504 & ManualBreak <= 4.058 ----- class = TOT_slow
    {
        'name': 'Rule1',
        'ubs': {'MPH': 0.865, 'FixationStart': -1.91, 'ManualBrake': 4.058},
        'lbs':{'PupilRight': -0.504},
        'out': 'slow'
    },
    # MPH <= 0.865 & FixationStart <= -1.663 & FixationStart <= -1.91 & PupilRight <= -0.504 & MPH <= -4.832 ----- class = TOT_slow
    {
        'name': 'Rule2',
        'ubs': {'MPH': -4.832, 'FixationStart': -1.91, 'PupilRight': -0.504},
        'lbs': {},
        'out': 'slow'
    },
    # MPH <= 0.865 & FixationStart <= -1.663 & FixationStart <= -1.91 & PupilRight <= -0.504 & MPH > -4.832 ----- class = TOT_med_slow
    {
        'name': 'Rule3',
        'ubs': {'MPH': 0.865, 'FixationStart': -1.91, 'PupilRight': -0.504},
        'lbs': {'MPH': -4.832},
        'out': 'med_slow'
    },
    # MPH > 0.865 and FizationStart > -0.17 and GazeDirectionRightZ ≤ 0.503 and InterpolatedGazeY > -0.047 and MPH <= 1.191 and PupilRight > -0.681 ------ class = TOT_med_fast
    {
        'name': 'Rule4',
        'ubs': {'GazeDirectionRightZ': 0.503, 'MPH': 1.191, },
        'lbs': {'MPH': 0.865, 'FixationStart': -0.17, 'InterpolatedGazeY': -0.047, 'PupilRight': -0.681},
        'out': 'med_fast'
    }
]

def get_initial_bounds_based_on_rule_inputs(X, features, rule):
    feature_names = features.keys()
    df = pd.DataFrame(X, columns=feature_names)
    # filter rows above rule's upper bounds
    for f,val in rule['ubs'].items():
        df = df[df[f] <= val]
    # filter rows below rule's lower bounds
    for f,val in rule['lbs'].items():
        df = df[df[f] >= val]
    # get initial lower bounds and upper bounds
    lower = df.min().to_numpy()
    upper = df.max().to_numpy()
    return lower, upper

def get_initial_bounds_based_on_rule_label(X, Y, labels, rule):
    label_index = labels[rule['out']]
    output = np.zeros(Y.shape[1]).astype(int)
    output[label_index] = 1
    indexes = np.where((Y == output).all(axis=1))[0]
    lower = [X[indexes][:, i].min() for i in range(X.shape[1])]
    upper = [X[indexes][:, i].max() for i in range(X.shape[1])]
    return lower, upper

def test_rule(net, rule, lower_bounds, upper_bounds, features, labels, categorical_feature_combos):
    target = labels[rule['out']]

    for y in range(5):
        if y == target: continue

        for cf in categorical_feature_combos:
            input_vars = net.inputVars[0].flatten()
            output_vars = net.outputVars[0].flatten()
            # set upper and lower bounds for all features (min/max values from training set)
            for i,v in enumerate(input_vars):
                net.setLowerBound(v, lower_bounds[i])
                net.setUpperBound(v, upper_bounds[i])
            # set categorical feature bounds
            for i,val in cf:
                net.setLowerBound(input_vars[i], val)
                net.setUpperBound(input_vars[i], val)
            # set rule bounds
            for f,val in rule['ubs'].items():
                net.setUpperBound(input_vars[features[f]], val)
            for f,val in rule['lbs'].items():
                net.setLowerBound(input_vars[features[f]], val)
            # add output query
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
            eq.addAddend(1, output_vars[y])
            eq.addAddend(-1, output_vars[target])
            eq.setScalar(0)
            net.addEquation(eq)
            # solve query
            vals, stats = net.solve(
                verbose=False,
                options=Marabou.createOptions(solveWithMILP=True, milpTightening='none')
                )
            if any(vals):
                inputs = [vals[v] for v in input_vars]
                outputs = [vals[v] for v in output_vars]
                return 'SAT', (inputs, outputs)
    return 'UNSAT', None

if __name__ == '__main__':

    @scriptify
    def script(conf_name='d13', #
               query='simple', # or 'complex'
               model_type='base', # or 'new'
               rules_type=25): # or 24

        data_path = f'../transfer_learning/data/'
        # model_path = '../network/models/v3.2.2/model.nnet'
        model_path = f'../transfer_learning/models/{conf_name}/model_{model_type}.nnet'
        tf_model_path = f'../transfer_learning/models/{conf_name}/model_{model_type}'
        marabou_logs_path = f'./marabou_logs/{conf_name}/{model_type}'
        if not os.path.exists(marabou_logs_path):
            os.makedirs(marabou_logs_path)
        features = pickle.load(open('./features.p', 'rb'))
        labels = pickle.load(open('./labels.p', 'rb'))

        df = pd.read_csv(f'{data_path}/train_full.csv', index_col=0)
        lower_bounds = df.iloc[:, 0:25].min().to_numpy()
        upper_bounds = df.iloc[:, 0:25].max().to_numpy()

        net = TOTNetV1(
            network_path=model_path,
            marabou_verbosity=0,
            marabou_options=dict(verbosity=1)#solveWithMILP=True, milpTightening='none')
            )

        rules = None
        if rules_type == 25:
            rules = rules_25
        elif rules_type == 24:
            rules = rules_24

        if query == 'simple':
            for r in rules:
                lbs, ubs = lower_bounds.copy(), upper_bounds.copy()
                for f,val in r['lbs'].items():
                    lbs[features.index(f)] = val
                for f,val in r['ubs'].items():
                    ubs[features.index(f)] = val
                y = labels.index(r['out'])
                pred, cex = net.find_counterexample(lbs, ubs, y,
                                                    filename=f'{marabou_logs_path}/{r["name"]}_{rules_type}.txt')
                result = 'UNSAT' if cex is None else 'SAT'
                print(f'{r["name"]} - class:{r["out"]}, pred:{labels[pred]}, result:{result}')
                if result == 'SAT':
                    print(cex)

        elif query == 'complex':
            X = pickle.load(open(f'{data_path}/X_train.p', 'rb'))
            Y = pickle.load(open(f'{data_path}/Y_train.p', 'rb'))
            features = {f:i for i,f in enumerate(pickle.load(open('./features.p', 'rb')))}
            # features = pickle.load(open('./features.p', 'rb'))
            labels = {l:i for i,l in enumerate(pickle.load(open('./labels.p', 'rb')))}
            # categorical_features = {
            #     22: (-8.516181955122368, -5.615958110066327, -2.7157342650102865, 0.1844895800457542),
            #     24: (-5.805253562105785, -0.1623221645152569, 5.480609233075271, 11.1235406306658)
            #     }
            # categorical_feature_combos = tuple(product(*[tuple(product((f,),fvals)) for f,fvals in categorical_features.items()]))
            categorical_feature_combos = (
                ((22, -8.516181955122368), (24, -5.805253562105785)),
                ((22, -8.516181955122368), (24, 11.1235406306658)),
                ((22, -5.615958110066327), (24, 5.480609233075271)),
                ((22, -5.615958110066327), (24, 11.1235406306658)),
                ((22, -2.7157342650102865), (24, -0.1623221645152569)),
                ((22, 0.1844895800457542), (24, -0.1623221645152569))
                )
            lower_bounds = [X[:, i].min() for i in range(X.shape[1])]
            upper_bounds = [X[:, i].max() for i in range(X.shape[1])]
            net = Marabou.read_tf(tf_model_path, modelType='savedModel_v2')


            for rule in rules:
                # test rule with generic initial bounds
                result, counterexample = test_rule(net, rule, lower_bounds, upper_bounds,
                                                   features, labels, categorical_feature_combos)
                # print(rule['name'] + '(generic)' + f' - class:' + str(rule['out']) + ', pred:' + str(np.argmax(counterexample[1])) + ', result:' + result)
                pred = str(np.argmax(counterexample[1])) if counterexample is not None else str(rule['out'])
                print(rule['name'] + '(generic)' + f' - class:' + str(rule['out']) + ', pred:' + pred + ', result:' + result)
                if result == 'SAT':
                    print('counterexample: ', counterexample)
                print('')

                # test rule with initial bounds based on rows matching rule
                lower, upper = get_initial_bounds_based_on_rule_inputs(X, features, rule)
                result, counterexample = test_rule(net, rule, lower, upper, features, labels, categorical_feature_combos)
                # print(rule['name'] + '(matching-rows)' + f' - class:' + str(rule['out']) + ', pred:' + str(np.argmax(counterexample[1])) + ', result:' + result)
                pred = str(np.argmax(counterexample[1])) if counterexample is not None else str(rule['out'])
                print(rule['name'] + '(matching-inputs)' + f' - class:' + str(rule['out']) + ', pred:' + pred + ', result:' + result)
                if result == 'SAT':
                    print('counterexample: ', counterexample)
                else:
                    print('UNSAT - BOUNDS:', lower, upper)
                print('')

                # test rule with initial bounds based on rows matching rule
                lower, upper = get_initial_bounds_based_on_rule_label(X, Y, labels, rule)
                result, counterexample = test_rule(rule, lower, upper, features, labels, categorical_feature_combos)
                # print(rule['name'] + '(matching-labels)' + f' - class:' + str(rule['out']) + ', pred:' + str(np.argmax(counterexample[1])) + ', result:' + result)
                pred = str(np.argmax(counterexample[1])) if counterexample is not None else str(rule['out'])
                print(rule['name'] + '(matching-labels)' + f' - class:' + str(rule['out']) + ', pred:' + pred + ', result:' + result)
                if result == 'SAT':
                    print('counterexample: ', counterexample)
                else:
                    print('UNSAT - BOUNDS:', lower, upper)
                print('-' * 60)




