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


def rule_specific_data(X_test, y_test, pred):

  feats = list(X_test.columns)

  test_x_arr = np.asarray(X_test)
  test_y_arr = np.asarray(y_test)
  pred = np.asarray(pred)

  rule1_lst = []
  for dpoint in range(len(X_test)):
      tmp1_lst = []
      if (X_test.iloc[dpoint, feats.index('ManualWheel')] <= 0.307) and (X_test.iloc[dpoint, feats.index('FixationStart')] <= -1.677) and (X_test.iloc[dpoint, feats.index('MPH')] <= -1.34):
        tmp1_lst.extend(test_x_arr[dpoint])
        tmp1_lst.extend(test_y_arr[dpoint])
        tmp1_lst.append('TOT_med')
        rule1_lst.append(tmp1_lst)
  print("Number of datapoints satisfying the rule: " + str(len(rule1_lst)))
  new_feats = feats + ['Target'] + ['Prediction']
  rule_df = pd.DataFrame(np.asarray(rule1_lst), columns = new_feats)
  acc_rule = sklearn.metrics.accuracy_score(rule_df['Target'], rule_df['Prediction'])
  print("Percentage of datapoints satisfying the rule classified correctly: " + str(acc_rule*100) + "%")


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    paths_pure = []
    paths_impure = []
    path = []

    def recurse(node, path, paths_pure, paths_impure):
        if tree_.children_left[node] != tree_.children_right[node]: #Internal node
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths_pure, paths_impure)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths_pure, paths_impure)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
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

    def gen_rules(paths):
        rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: " + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]
        return rules

    rules_pure = gen_rules(paths_pure)
    rules_impure = gen_rules(paths_impure)

    return (rules_pure,rules_impure)


if __name__ == '__main__':

    @scriptify
    def script(conf_name='default',
               gpu=0):
        " Setup & Install """
        # Some global variables and general settings

        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        saved_model_dir = f'./models/{conf_name}'
        data_dir = f'./data/{conf_name}'
        rules_dir = f'./rules/{conf_name}'
        if not os.path.exists(rules_dir):
            os.makedirs(rules_dir)

        neural_model = load_model(f'{saved_model_dir}/model_base')
        neural_model.summary()

        df_train = pd.read_csv(f'{data_dir}/train_base.csv')
        X_train = df_train.iloc[:, 1:26]
        Y_train = df_train.iloc[:, 26:]
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

        (rules_pure, rules_impure) = get_rules(clf,feature_names,y_labels)
        file_rules_pure = f'{rules_dir}/rules_pure.txt'
        with open(file_rules_pure, 'w') as f1:
            for rule in rules_pure:
                f1.write(rule)
                f1.write("\n")
        file_rules_impure = f'{rules_dir}/rules_impure.txt'
        with open(file_rules_impure, 'w') as f2:
            for rule in rules_impure:
                f2.write(rule)
                f2.write("\n")

