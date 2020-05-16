import os, logging, random, time
import pandas as pd
from decimal import Decimal
from tot_net import TOTNet
from tensorflow.keras.models import load_model

def count_decimal_places(f):
    return abs(Decimal(str(f)).as_tuple().exponent)

def ms_since_1970():
    return int(round(time.time() * 1000))

def create_logger(name, level=logging.DEBUG, to_file=False, to_console=True, logpath='logs'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if to_file:
        filename = os.path.join(logpath, f'{name}.log')
        fh = logging.FileHandler(filename=filename)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger

def print_heading(text):
    print(('-'*40) + f'\n{text}\n')

class TOTUtils:
    _features = ('FixationDuration', 'FixationSeq', 'FixationStart', 'FixationX', 
                 'FixationY', 'GazeDirectionLeftZ', 'GazeDirectionRightZ', 'PupilLeft', 
                 'PupilRight', 'InterpolatedGazeX', 'InterpolatedGazeY', 'AutoThrottle', 
                 'AutoWheel', 'CurrentThrottle', 'CurrentWheel', 'Distance3D', 'MPH', 
                 'ManualBrake', 'ManualThrottle', 'ManualWheel', 'RangeW', 'RightLaneDist', 
                 'RightLaneType', 'LeftLaneDist', 'LeftLaneType')
    _categories = ('fast', 'med_fast', 'med', 'med_slow', 'slow')

    @staticmethod
    def get_feature_names():
        return list(TOTUtils._features)

    @staticmethod
    def get_feature_name(input_x):
        return TOTUtils._features[input_x]
    
    @staticmethod
    def get_feature_index(feature_name):
        return TOTUtils._features.index(feature_name)
    
    @staticmethod
    def get_category_index(label):
        return TOTUtils._categories.index(label)
    
    @staticmethod
    def get_majority_category_index(output_sample):
        return output_sample.index(max(output_sample))

    @staticmethod
    def get_scaled_value(scikit_scaler,  col_idx, actual_value):
        '''
        returns the scaled value of a single column
        '''
        dummy = [[0] * len(scikit_scaler.scale_)]
        dummy[0][col_idx] = actual_value
        return scikit_scaler.transform(dummy)[0][col_idx]

    @staticmethod
    def get_actual_value(scikit_scaler, col_idx, scaled_value):
        '''
        returns the actual value of a single column
        '''
        dummy = [[0] * len(scikit_scaler.scale_)]
        dummy[0][col_idx] = scaled_value
        return scikit_scaler.inverse_transform(dummy)[0][col_idx] 

    @staticmethod
    def get_scaled_values(scikit_scaler, actual_values):
        '''
        returns the scaled values
        '''
        assert(len(scikit_scaler.scale_) == len(actual_values))
        return scikit_scaler.transform(actual_values)[0]

    @staticmethod
    def get_actual_values(scikit_scaler, scaled_values):
        '''
        returns the scaled value of a single column from
        '''
        assert(len(scikit_scaler.scale_) == len(scaled_values))
        return scikit_scaler.inverse_transform(scaled_values)[0]

    @staticmethod
    def load_samples(csv_path, frac=1):
        '''
        loads a csv, and converts to a list of samples [([x0-xN], [y0..yM]), ([x0..xM], [y0..yM])]

        @param csv_path (string): path to csv file
        @frac (float): fraction of dataset to return (default=1.0)
        '''
        df = pd.read_csv(csv_path, index_col=0)
        df = df.sample(frac=frac)
        output_cols = [f'TOT_{c}' for c in TOTUtils._categories]
        outputs_df = df[output_cols]
        inputs_df = df.drop(output_cols, axis=1)
        return [(list(inputs_df.iloc[i]), list(outputs_df.iloc[i])) for i in range(df.shape[0])]

    @staticmethod
    def filter_samples(samples, model_path, incorrect_predictions=False):
        '''
        evaluates samples and filters based on correct or incorrect prediction

        @param samples (list): list of tuples containing inputs and outputs
        @param model_path (string): path to h5 or pb model
        @incorrect_predictions (bool): if true, returns the incorrect predictions (default False)
        '''
        filtered = []
        model = load_model(model_path)
        input_samples = [s[0] for s in samples]
        output_samples = [s[1] for s in samples]
        predictions = model.predict(input_samples)
        for i,p in enumerate(predictions):
            inputs, outputs = input_samples[i], output_samples[i]
            exp_cat = outputs.index(max(outputs))
            correct = list(p).index(max(p)) == exp_cat
            if (not incorrect_predictions and correct) or (incorrect_predictions and not correct):
                filtered.append((inputs, outputs))
        return filtered
    
    @staticmethod
    def save_samples_to_csv(samples, output_file):
        '''
        saves samples to a single csv file

        @param samples (list): list of input samples (tuples contianing inputs & outputs)
        @param output_file (string): output file path
        '''
        df = pd.concat([
            pd.DataFrame([s[0] for s in samples], columns=TOTUtils._features),
            pd.DataFrame([s[1] for s in samples], columns=[f'TOT_{c}' for c in TOTUtils._categories])
            ], axis=1)
        df.to_csv(output_file)
        