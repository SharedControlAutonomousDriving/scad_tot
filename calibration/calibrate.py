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


if __name__ == '__main__':

    @scriptify
    def script():
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