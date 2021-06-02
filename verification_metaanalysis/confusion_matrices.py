import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model

# CALIBRATION LIBRARIES
import tensorflow_probability as tfp
from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling
from netcal.presentation import ReliabilityDiagram
from scriptify import scriptify
from scipy.spatial import distance

num_verification_methods = 2
not_verifiedreg_and_not_confident = 0
not_verifiedreg_and_confident = 1
verifiedreg_and_not_confident = 2
verifiedreg_and_confident = 3
num_classes = 5

def isInVerifiedRegion(x, centroids, l2_radii):
    for (centroid, l2_radius) in zip(centroids,l2_radii):
        if distance.euclidean(x, centroid) <= l2_radius:
            return True
    return False

def updateConfusionMatrix(conf_matrix, act_label, pred_label):
    conf_matrix[act_label][pred_label] += 1

if __name__ == '__main__':

    @scriptify
    def script(experiment="confusion",
               conf_threshold=0.7):
        '''
                    LOAD DATA
                    '''
        data_path = '../data/'
        train_data = pd.read_csv(data_path + 'v3.2.2_train.csv')
        test_data = pd.read_csv(data_path + 'v3.2.2_test.csv')
        test_data_splits = np.array_split(test_data, 2)

        y_train, y_test = train_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']], \
                          test_data_splits[0][['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        X_train, X_test = train_data.drop(
            ['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1), \
                          test_data_splits[0].drop(
                              ['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        model = load_model("../network/models/v3.2.2/model.h5")
        print("# of train samples: ", len(train_data.index))
        print("# of test samples: ", len(test_data.index))
        print("# of test samples held out for verifying probabilistic model: ", len(test_data_splits[1].index))

        calibrated_data = pd.read_csv('../calibration/calibration-data/test_calibrated_v3.2.2.csv', header=None)
        verified_regions_data = pd.read_csv('../data/vregions_combined.csv')

        y_pred = model.predict(X_test.values)

        centroids = []
        l2_radii = []
        for index, row in verified_regions_data.iterrows():
            l2_radii.append(row['radius'])
            centroids.append(np.array([row['cx0'], row['cx1'], row['cx2'], row['cx3'], row['cx4'],
                                       row['cx5'], row['cx6'], row['cx7'], row['cx8'], row['cx9'],
                                       row['cx10'], row['cx11'], row['cx12'], row['cx13'], row['cx14'],
                                       row['cx15'], row['cx16'], row['cx17'], row['cx18'], row['cx19'],
                                       row['cx20'], row['cx21'], row['cx22'], row['cx23'], row['cx24']]))

        if experiment == "confidence":

            num_in_verified_regions = 0
            num_acc_in_verified_regions = 0
            num_not_in_verified_regions = {'0.0': 0, '0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0,
                                                    '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 0, '0.9': 0}

            num_acc_not_in_verified_regions = {'0.0': 0, '0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0,
                                                    '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 0, '0.9': 0}
            for index, row in X_test.iterrows():
                y_pred_i = y_pred[index,:]
                acc_i = (np.argmax(y_pred_i) == np.argmax(y_test.iloc[index].values))
                y_calib_i = calibrated_data.iloc[index].values
                if isInVerifiedRegion(row.values, centroids, l2_radii):
                    num_in_verified_regions += 1
                    if acc_i:
                        num_acc_in_verified_regions += 1
                else:
                    y_conf_i = y_calib_i[np.argmax(y_pred_i)]
                    if (y_conf_i >= 0.9):
                        num_not_in_verified_regions['0.9'] += 1
                    elif (y_conf_i >= 0.8):
                        num_not_in_verified_regions['0.8'] += 1
                    elif (y_conf_i >= 0.7):
                        num_not_in_verified_regions['0.7'] += 1
                    elif (y_conf_i >= 0.6):
                        num_not_in_verified_regions['0.6'] += 1
                    elif (y_conf_i >= 0.5):
                        num_not_in_verified_regions['0.5'] += 1
                    elif (y_conf_i >= 0.4):
                        num_not_in_verified_regions['0.4'] += 1
                    elif (y_conf_i >= 0.3):
                        num_not_in_verified_regions['0.3'] += 1
                    elif (y_conf_i >= 0.2):
                        num_not_in_verified_regions['0.2'] += 1
                    elif (y_conf_i >= 0.1):
                        num_not_in_verified_regions['0.1'] += 1
                    else:
                        num_not_in_verified_regions['0.0'] += 1

                    if acc_i:
                        if (y_conf_i >= 0.9):
                            num_acc_not_in_verified_regions['0.9'] += 1
                        elif (y_conf_i >= 0.8):
                            num_acc_not_in_verified_regions['0.8'] += 1
                        elif (y_conf_i >= 0.7):
                            num_acc_not_in_verified_regions['0.7'] += 1
                        elif (y_conf_i >= 0.6):
                            num_acc_not_in_verified_regions['0.6'] += 1
                        elif (y_conf_i >= 0.5):
                            num_acc_not_in_verified_regions['0.5'] += 1
                        elif (y_conf_i >= 0.4):
                            num_acc_not_in_verified_regions['0.4'] += 1
                        elif (y_conf_i >= 0.3):
                            num_acc_not_in_verified_regions['0.3'] += 1
                        elif (y_conf_i >= 0.2):
                            num_acc_not_in_verified_regions['0.2'] += 1
                        elif (y_conf_i >= 0.1):
                            num_acc_not_in_verified_regions['0.1'] += 1
                        else:
                            num_acc_not_in_verified_regions['0.0'] += 1

            print("# of test samples in verified regions: ", num_in_verified_regions)
            print("# of correctly predicted test samples in verified regions: ",
                  num_acc_in_verified_regions)
            print("# of test samples NOT in verified regions: ", num_not_in_verified_regions)
            print("# of correctly predicted test samples NOT in verified regions: ",
                  num_acc_not_in_verified_regions)

        elif experiment == "confusion":
            num_verified = np.zeros((2 ** num_verification_methods))
            confusion_matrices = []
            for i in range(0, 2 ** num_verification_methods):
                confusion_matrices.append(np.zeros((num_classes, num_classes)))

            for index, row in X_test.iterrows():
                y_pred_i = y_pred[index,:]
                pred_label_i = np.argmax(y_pred_i)
                act_label_i = np.argmax(y_test.iloc[index].values)
                y_calib_i = calibrated_data.iloc[index].values
                y_conf_i = y_calib_i[np.argmax(y_pred_i)]

                inVerifiedRegion = isInVerifiedRegion(row.values, centroids, l2_radii)
                confident = y_conf_i >= conf_threshold

                if inVerifiedRegion:
                    if confident:
                        num_verified[verifiedreg_and_confident] += 1
                        updateConfusionMatrix(confusion_matrices[verifiedreg_and_confident], act_label_i, pred_label_i)
                    else:
                        num_verified[verifiedreg_and_not_confident] += 1
                        updateConfusionMatrix(confusion_matrices[verifiedreg_and_not_confident], act_label_i, pred_label_i)
                else:
                    if confident:
                        num_verified[not_verifiedreg_and_confident] += 1
                        updateConfusionMatrix(confusion_matrices[not_verifiedreg_and_confident], act_label_i, pred_label_i)
                    else:
                        num_verified[not_verifiedreg_and_not_confident] += 1
                        updateConfusionMatrix(confusion_matrices[not_verifiedreg_and_not_confident], act_label_i, pred_label_i)

            print("Results for p=", conf_threshold)
            print("# of test samples in each combination of verification outcomes: ", num_verified)
            print("confusion matrix for not in a verified region and not confident: ", confusion_matrices[not_verifiedreg_and_not_confident])
            print("confusion matrix for not in a verified region and confident: ", confusion_matrices[not_verifiedreg_and_confident])
            print("confusion matrix for in a verified region and not confident: ", confusion_matrices[verifiedreg_and_not_confident])
            print("confusion matrix for in a verified region and confident: ", confusion_matrices[verifiedreg_and_confident])



