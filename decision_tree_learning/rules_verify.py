import pickle
import pandas as pd
from verification.tot_net import TOTNetV1
import pickle
import numpy as np
import pandas as pd
from itertools import product
from maraboupy import Marabou, MarabouCore, MarabouUtils
from scriptify import scriptify

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

def test_rule(rule, lower_bounds, upper_bounds):
    target = labels[rule['out']]

    for y in range(5):
        if y == target: continue

        for cf in categorical_feature_combos:
            net = Marabou.read_tf(
                '../network/models/v3.2.2/model-verification',
                modelType='savedModel_v2'
                )
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
    def script(epochs=30,
               batch_size=128,
               dataset_file='All_Features_ReactionTime.csv',
               conf_name='default',
               new_driver_ids='013_M1;013_M2;013_M3'):

    # model_path = '../network/models/v3.2.2/model.nnet'
    model_path = '../network/models/v3.2.2/model-verification'
    features = pickle.load(open('../data/v3.2.2/features.p', 'rb'))
    labels = pickle.load(open('../data/v3.2.2/labels.p', 'rb'))
    # X, Y = pickle.load(open('../data/v3.2.2/verification.p', 'rb'))

    df = pd.read_csv('../data/v3.2.2/train.csv', index_col=0)
    lower_bounds = df.iloc[:, 0:25].min().to_numpy()
    upper_bounds = df.iloc[:, 0:25].max().to_numpy()

    # RULE 1: (med_fast)
    # ManualWheel<=0.307 & FixationStart<=-1.677 & MPH>-1.34
    rule1 = {
        'name': 'rule1',
        'ubs': {'ManualWheel': 0.307, 'FixationStart':-1.677},
        'lbs':{'MPH':-1.34},
        'out': 'med_fast'
        }

    # RULE 2: (med_slow)
    # ManualWheel<=0.307 & FixationStart<=-1.677 & MPH<=-1.34 & FixationSeq>-1.704
    rule2 = {
        'name': 'rule2',
        'ubs': {'ManualWheel': 0.307, 'FixationStart':-1.677, 'MPH': -1.34},
        'lbs':{'FixationSeq': -1.704},
        'out': 'med_slow'
        }

    # RULE 3: (med_fast)
    # ManualWheel<=0.307 & FixationStart<=-1.677 & MPH<=-1.34 & FixationSeq<=-1.704 & MPH<=-2.126
    rule3 = {
        'name': 'rule3',
        # 'ubs': {'ManualWheel': 0.307, 'FixationStart':-1.677, 'MPH': -1.34, 'FixationSeq': -1.704, 'MPH':-2.126},
        'ubs': {'ManualWheel': 0.307, 'FixationStart':-1.677, 'MPH': -1.34, 'FixationSeq': -1.704, 'MPH':-2.126},
        'lbs':{},
        'out': 'med_fast'
        }

    # RULE 4: (med)
    # ManualWheel<=0.307 & FixationStart<=-1.677 & MPH<=-1.34 & FixationSeq<=-1.704 & MPH>-2.126
    rule4 = {
        'name': 'rule4',
        'ubs': {'ManualWheel': 0.307, 'FixationStart':-1.677, 'MPH': -1.34, 'FixationSeq': -1.704},
        'lbs':{'MPH':-2.126},
        'out': 'med'
        }

    rules = [rule1, rule2, rule3, rule4]

    net = TOTNetV1(
        network_path=model_path,
        network_options=dict(modelType='savedModel_v2'),
        marabou_verbosity=0,
        marabou_options=dict(solveWithMILP=True, milpTightening='none')
        )

    for r in rules:
        lbs, ubs = lower_bounds.copy(), upper_bounds.copy()
        for f,val in r['lbs'].items():
            lbs[features.index(f)] = val
        for f,val in r['ubs'].items():
            ubs[features.index(f)] = val
        y = labels.index(r['out'])
        pred, cex = net.find_counterexample(lbs, ubs, y)
        result = 'UNSAT' if cex is None else 'SAT'
        print(f'{r["name"]} - class:{r["out"]}, pred:{labels[pred]}, result:{result}')
        if result == 'SAT':
            print(cex)


    # In[15]:

    # 1. ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH > -1.34 : class = TOT_med
    # 2. ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence > -1.704 : class = TOT_med_slow
    # 3. ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH <= -2.126 : CLASS = TOT_med
    # 4. ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH > -2.126 : CLASS = TOT_med_fast
    # 5. ManualWheel <= 0.307 & FixationStart > -1.677 & FixationStart > 1.717 & PupilLeft > 0.424 & FixationStart > 1.924 : CLASS = TOT_slow
    # 6. ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel <= 0.887 : CLASS: TOT_med_slow
    # 7. ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel > 0.887 : CLASS: TOT_fast

    rules = [
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


    model_path = '../network/models/v3.2.2/model-verification'
    X = pickle.load(open('../data/v3.2.2/X_train.p', 'rb'))
    Y = pickle.load(open('../data/v3.2.2/Y_train.p', 'rb'))
    features = {f:i for i,f in enumerate(pickle.load(open('../data/v3.2.2/features.p', 'rb')))}
    # features = pickle.load(open('../data/v3.2.2/features.p', 'rb'))
    labels = {l:i for i,l in enumerate(pickle.load(open('../data/v3.2.2/labels.p', 'rb')))}
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



    for rule in rules:
        # test rule with generic initial bounds
        result, counterexample = test_rule(rule, lower_bounds, upper_bounds)
        # print(rule['name'] + '(generic)' + f' - class:' + str(rule['out']) + ', pred:' + str(np.argmax(counterexample[1])) + ', result:' + result)
        pred = str(np.argmax(counterexample[1])) if counterexample is not None else str(rule['out'])
        print(rule['name'] + '(generic)' + f' - class:' + str(rule['out']) + ', pred:' + pred + ', result:' + result)
        if result == 'SAT':
            print('counterexample: ', counterexample)
        print('')

        # test rule with initial bounds based on rows matching rule
        lower, upper = get_initial_bounds_based_on_rule_inputs(X, features, rule)
        result, counterexample = test_rule(rule, lower, upper)
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
        result, counterexample = test_rule(rule, lower, upper)
        # print(rule['name'] + '(matching-labels)' + f' - class:' + str(rule['out']) + ', pred:' + str(np.argmax(counterexample[1])) + ', result:' + result)
        pred = str(np.argmax(counterexample[1])) if counterexample is not None else str(rule['out'])
        print(rule['name'] + '(matching-labels)' + f' - class:' + str(rule['out']) + ', pred:' + pred + ', result:' + result)
        if result == 'SAT':
            print('counterexample: ', counterexample)
        else:
            print('UNSAT - BOUNDS:', lower, upper)
        print('-' * 60)


    # In[16]:





    # In[ ]:


    # ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence > -1.704 : class = TOT_med_slow
    # ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH <= -2.126 : CLASS = TOT_med

    # ManualWheel <= 0.307 & FixationStart <= -1.677 & MPH <= -1.34 & FixationSequence <= -1.704 & MPH > -2.126 : CLASS = TOT_med_fast

    # ManualWheel <= 0.307 & FixationStart > -1.677 & FixationStart > 1.717 & PupilLeft > 0.424 & FixationStart > 1.924 : CLASS = TOT_slow
    # ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel <= 0.887 : CLASS: TOT_med_slow
    # ManualWheel > 0.307 & ManualWheel <= 0.77 & ManualWheel <= 0.461 & InterpolatedGazeY > 0.109 & FixationSequence <= 0.62 & Distance3D > 0.283 & FixationSequence > -0.043 & AutoWheel > 0.887 : CLASS: TOT_fast

    new_rules = [
        {
        'name': 'Rule1',
        'ubs': {'ManualWheel': 0.307, 'FixationStart': -1.677, 'MPH': -1.34},
        'lbs':{'FixationSeq': -1.704},
        'out': 'med_slow'
        },
        {
        'name': 'Rule2',
        'ubs': {'ManualWheel': 0.307, 'FixationStart': -1.677, 'MPH': -2.126, 'FixationSeq': -1.704},
        'lbs':{},
        'out': 'med'
        },
        {
        'name': 'Rule3',
        'ubs': {'ManualWheel': 0.307, 'FixationStart': -1.677, 'MPH': -1.34, 'FixationSeq': -1.704},
        'lbs':{'MPH':-2.126},
        'out': 'med_fast'
        }
    ]


    # In[ ]:



