import pandas as pd
import numpy as np
import sklearn
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from scriptify import scriptify
import os
from tensorflow.keras.models import Sequential, load_model


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


if __name__ == '__main__':

    @scriptify
    def script(conf_name='default'):
        " Setup & Install """
        # Some global variables and general settings
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
        y_label = list(Y_train.columns)
        Y_train_pred_onehot = neural_model.predict(X_train)
        lst_labels = []
        for pred in Y_train_pred_onehot:
            loc = np.where(pred == 1)[0]
            lst_labels.append(y_label[loc])
        # Target values ['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']
        Y_train_pred = DataFrame(lst_labels, columns=['Target'])

        df_test = pd.read_csv(f'{data_dir}/test_base.csv')
        X_test = df_test.iloc[:, 1:26]
        Y_test = df_test.iloc[:, 26:]
        Y_test_pred_onehot = neural_model.predict(X_test)
        lst_labels = []
        for pred in Y_test_pred_onehot:
            loc = np.where(pred == 1)[0]
            lst_labels.append(y_label[loc])
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
        tree_train_acc = sklearn.metrics.accuracy_score(pred, Y_train_pred)
        tree_test_acc = sklearn.metrics.accuracy_score(pred, Y_test_pred)
        print("Decision tree train accuracy wrt to neural model: ", tree_train_acc)
        print("Decision tree test accuracy wrt to neural model: ", tree_test_acc)
