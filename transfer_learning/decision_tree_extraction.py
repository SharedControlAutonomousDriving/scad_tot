import pandas as pd
import numpy as np
import sklearn
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from sklearn.tree import export_text
from scriptify import scriptify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


def get_paths(tree, impurity_threshold):
    tree_ = tree.tree_
    paths_pure = []
    paths_impure = []
    path = []

    def recurse(node, path, paths_pure, paths_impure):
        if tree_.children_left[node] != tree_.children_right[node]: #Internal node
            # name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            # p1 += [f"({name} <= {np.round(threshold, 3)})"]
            p1 += [("ub", tree_.feature[node], np.round(threshold, 3))]
            recurse(tree_.children_left[node], p1, paths_pure, paths_impure)
            # p2 += [f"({name} > {np.round(threshold, 3)})"]
            p2 += [("lb", tree_.feature[node], np.round(threshold, 3))]
            recurse(tree_.children_right[node], p2, paths_pure, paths_impure)
        else:
            classes = tree_.value[node][0]
            l = np.argmax(classes)
            prob = np.round(100.0 * classes[l] / np.sum(classes), 2)
            path += [(tree_.value[node], tree_.n_node_samples[node], prob)]
            if tree_.impurity[node] == 0:
                paths_pure += [path]
            else:
                paths_impure += [path]

    recurse(0, path, paths_pure, paths_impure)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths_pure]
    ii = list(np.argsort(samples_count))
    paths_pure = [paths_pure[i] for i in reversed(ii)]

    samples_count = [p[-1][1] for p in paths_impure]
    ii = list(np.argsort(samples_count))
    paths_impure = [paths_impure[i] for i in reversed(ii)]
    paths_impure = list(filter((lambda p: p[-1][2] >= impurity_threshold), paths_impure))

    return (paths_pure,paths_impure)


def print_rules(paths, feature_names, class_names, file_path):
    with open(file_path, 'w') as f1:
        for path in paths:
            rule = "if "
            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += f'{feature_names[p[1]]} <= {p[2]}' if p[0] == "lb" else f'{feature_names[p[1]]} > {p[2]}'
            rule += " then "
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {path[-1][2]}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            f1.write(rule)
            f1.write("\n")


def print_marabou_query(path, lbs, ubs, file_path):
    lbs, ubs = lbs.copy(), ubs.copy()
    for p in path[:-1]:
        if p[0] == "lb":
            lbs[p[1]] = p[2] if p[2] > lbs[p[1]] else lbs[p[1]]
        else:
            ubs[p[1]] = p[2] if p[2] < ubs[p[1]] else ubs[p[1]]

    correct_label = np.argmax(path[-1][0][0])
    incorrect_labels = list(filter(lambda l: l != correct_label, [0,1,2,3,4]))
    names = ['a','b','c','d']

    for i in range(0, len(incorrect_labels)):
        fpath = f'{file_path}{names[i]}.txt'
        with open(fpath, 'w') as f1:
            for j in range(0, lbs.size):
                f1.write(f'x{j} >= {lbs[j]}\n')
                f1.write(f'x{j} < {ubs[j]}\n')
            f1.write(f'y{incorrect_labels[i]} -y{correct_label} >= 0')


if __name__ == '__main__':

    @scriptify
    def script(conf_name='default',
               impurity=80, #purity percentage threshold for choosing an impure rule,
               num_queries=5, #number of Marabou queries to print
               gpu=0):
        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        # Some global variables and general settings
        saved_model_dir = f'./models/{conf_name}'
        data_dir = f'./data/{conf_name}'
        rules_dir = f'./rules/{conf_name}'
        if not os.path.exists(rules_dir):
            os.makedirs(rules_dir)

        neural_model = load_model(f'{saved_model_dir}/model_base')
        neural_model.summary()

        # Prepare data
        df_train = pd.read_csv(f'{data_dir}/train_base.csv', index_col=0)
        lower_bounds = df_train.iloc[:, 0:25].min().to_numpy()
        upper_bounds = df_train.iloc[:, 0:25].max().to_numpy()
        X_train = df_train.iloc[:, 0:25]
        Y_train = df_train.iloc[:, 25:]
        feature_names = list(X_train.columns)
        y_labels = list(Y_train.columns)
        Y_train_pred_onehot = neural_model.predict(X_train)
        lst_labels = []
        for pred in Y_train_pred_onehot:
            loc = np.argmax(pred)
            lst_labels.append(y_labels[loc])
        # Target values ['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']
        Y_train_pred = DataFrame(lst_labels, columns=['Target'])

        df_test = pd.read_csv(f'{data_dir}/test_base.csv')
        X_test = df_test.iloc[:, 1:26]
        Y_test = df_test.iloc[:, 26:]
        Y_test_pred_onehot = neural_model.predict(X_test)
        lst_labels = []
        for pred in Y_test_pred_onehot:
            loc = np.argmax(pred)
            lst_labels.append(y_labels[loc])
        # Target values ['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']
        Y_test_pred = DataFrame(lst_labels, columns=['Target'])


        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth = 10,
                                     random_state = 0)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,Y_train_pred)

        tree_train_pred = clf.predict(X_train)
        tree_test_pred = clf.predict(X_test)
        # accuracy of predictions using decision tree
        tree_train_acc = sklearn.metrics.accuracy_score(tree_train_pred, Y_train_pred)
        tree_test_acc = sklearn.metrics.accuracy_score(tree_test_pred, Y_test_pred)
        print("Decision tree train accuracy wrt to neural model: ", tree_train_acc)
        print("Decision tree test accuracy wrt to neural model: ", tree_test_acc)

        # Print rules and Marabou queries
        (paths_pure, paths_impure) = get_paths(clf, impurity)
        file_rules_pure = f'{rules_dir}/rules_pure.txt'
        print_rules(paths_pure, feature_names, y_labels, file_rules_pure)
        file_rules_impure = f'{rules_dir}/rules_impure.txt'
        print_rules(paths_impure, feature_names, y_labels, file_rules_impure)

        for cnt in range(0, num_queries):
            path = paths_pure[cnt]
            query_file_path = f'{rules_dir}/pure_q_{cnt}'
            print_marabou_query(path,lower_bounds,upper_bounds,query_file_path)

        for cnt in range(0, num_queries):
            path = paths_impure[cnt]
            query_file_path = f'{rules_dir}/impure_q_{cnt}'
            print_marabou_query(path, lower_bounds, upper_bounds, query_file_path)
            cnt += 1




