import os, logging, random, time
from pathlib import Path
import pandas as pd
from decimal import Decimal
from tot_net import TOTNet

def count_decimal_places(f):
    return abs(Decimal(str(f)).as_tuple().exponent)

def ms_since_1970():
    return int(round(time.time() * 1000))

def create_logger(name, level=logging.DEBUG, to_file=False, to_console=True, logdir='logs'):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if to_file:
        if not os.path.exists(logdir):
            os.makedirs(logdir, mode=0o755)
        filename = os.path.join(logdir, f'{name}.log')
        fh = logging.FileHandler(filename=filename)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger

def print_heading(text):
    print(('-'*40) + f'\n{text}\n')

def tohex(r, g, b):
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def create_dirpath(outpath, dirmode=0o755):
    # create directory path if doesn't exist.
    Path(outpath).mkdir(parents=True, exist_ok=True)

class TOTSample:
    def __init__(self, inputs, outputs, id=None):
        self.inputs = inputs
        self.outputs = outputs

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
        return TOTUtils.get_feature_names()[input_x]
    
    @staticmethod
    def get_feature_index(feature_name):
        return TOTUtils.get_feature_names().index(feature_name)
    
    @staticmethod
    def get_category_names():
        return list(TOTUtils._categories)

    @staticmethod
    def get_category_name(y_index):
        return TOTUtils.get_category_names()[y_index]
    
    @staticmethod
    def get_category_index(label):
        return TOTUtils.get_category_names().index(label)
    
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
    def load_csv(csv_path, frac=1):
        '''
        loads a csv, and returns two dataframes (inputs and outputs)

        @param csv_path (string): path to csv file
        @frac (float): fraction of dataset to return (default=1.0)
        '''
        df = pd.read_csv(csv_path, index_col=0)
        df = df.sample(frac=frac)
        output_cols = [f'TOT_{c}' for c in TOTUtils._categories]
        return df.drop(output_cols, axis=1), df[output_cols]

    @staticmethod
    def load_samples(csv_path, frac=1):
        '''
        loads a csv, and converts to a list of samples [([x0-xN], [y0..yM]), ([x0..xM], [y0..yM])]

        @param csv_path (string): path to csv file
        @frac (float): fraction of dataset to return (default=1.0)
        '''
        inputs_df, outputs_df = TOTUtils.load_csv(csv_path, frac=frac)
        return [(list(inputs_df.iloc[i]), list(outputs_df.iloc[i])) for i in range(inputs_df.shape[0])]

    @staticmethod
    def group_samples(samples):
        '''
        groups samples by category. returns a list containing samples for each category.
        '''
        n_outputs = len(samples[0][1])
        groups = [[] for i in range(n_outputs)]
        for sample in samples:
            _, outputs = sample
            y = outputs.index(max(outputs))
            groups[y].append(sample)
        return groups

    @staticmethod
    def filter_samples(samples, nnet_path, incorrect=False):
        '''
        evaluates samples and filters based on correct or incorrect prediction

        @param samples (list): list of tuples containing inputs and outputs
        @param nnet_path (string): path to nnet model
        @incorrect (bool): if true, returns the incorrect predictions (default False)
        '''
        filtered = []
        net = TOTNet(nnet_path)
        for sample in samples:
            inputs, outputs = sample
            y = outputs.index(max(outputs))
            is_correct = net.check_prediction(inputs, y)
            if (not incorrect and is_correct) or (incorrect and not is_correct):
                filtered.append(sample)
        return filtered
    
    @staticmethod
    def save_samples_to_csv(samples, outdir):
        '''
        saves samples to a single csv file

        @param samples (list): list of input samples (tuples contianing inputs & outputs)
        @param output_file (string): output file path
        '''
        if not os.path.exists(outdir):
            os.makedirs(outdir, mode=0o755)
        outfile = os.path.join(outdir, 'samples.csv')
        df = pd.concat([
            pd.DataFrame([s[0] for s in samples], columns=TOTUtils._features),
            pd.DataFrame([s[1] for s in samples], columns=[f'TOT_{c}' for c in TOTUtils._categories])
            ], axis=1)
        df.to_csv(outfile)
        return outfile
    
    @staticmethod
    def build_adversarial_samples(self, target_samples, outdir):
        pass
