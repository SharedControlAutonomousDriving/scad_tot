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


def isInVerifiedRegion(x, centroids, l2_radii):
    for (centroid, l2_radius) in zip(centroids,l2_radii):
        if distance.euclidean(x, centroid) <= l2_radius:
            return True
    return False



if __name__ == '__main__':

    @scriptify
    def script(experiment='calibrate'):
        '''
                    LOAD DATA
                    '''
        data_path = '../data/'
        train_data = pd.read_csv(data_path + 'v3.2.2_train.csv')
        test_data = pd.read_csv(data_path + 'v3.2.2_test.csv')

        y_train, y_test = train_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']], \
                          test_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        X_train, X_test = train_data.drop(
            ['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1), \
                          test_data.drop(
                              ['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        model = load_model("../network/models/v3.2.2/model.h5")
        print("# of train samples: ", len(y_train.index))
        print("# of test samples: ", len(y_test.index))

        if experiment == 'calibrate':

            ##Using NetCal package
            n_bins = 10
            confidences = model.predict(X_test.values)
            ece = ECE(n_bins)
            uncalibrated_score = ece.measure(confidences,y_test.values.argmax(axis=1))
            print("Calibration Error before calibration: ",uncalibrated_score)

            temperature = TemperatureScaling()
            temperature.fit(confidences, y_test.values.argmax(axis=1))
            calibrated = temperature.transform(confidences)
            ece = ECE(n_bins)
            calibrated_score = ece.measure(calibrated,y_test.values.argmax(axis=1))
            print("Calibration Error after calibration: ",calibrated_score)


            diagram = ReliabilityDiagram(n_bins)
            diagram.plot(confidences, y_test.values.argmax(axis=1))  # visualize miscalibration of uncalibrated

            diagram.plot(calibrated, y_test.values.argmax(axis=1))   # visualize miscalibration of calibrated

            np.savetxt('./calibration-data/test_calibrated_v3.2.2.csv',calibrated,delimiter=',')

        elif experiment == 'eval':
            calibrated_data = pd.read_csv('./calibration-data/test_calibrated_v3.2.2.csv', header=None)
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