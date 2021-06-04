import os, logging, random, time, math
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Union
from pathlib import Path
from decimal import Decimal
from scipy.special import softmax

# # TODO: remove tot_net import from utils
# from verification.tot_net import TOTNet

def chunk_dataset(X:np.array, Y:np.array, chunksize:int):
    n_chunks = math.ceil(X.shape[0] / chunksize)
    X_chunks = np.array_split(X, chunksize, axis=0)
    Y_chunks = np.array_split(Y, chunksize, axis=0)
    n_chunks = len(X_chunks)
    chunks = [(X_chunks[0], Y_chunks[0]) for i in range(n_chunks)]
    return chunks

def _parse_onehot_features(onehot_feature_defs:List[str]) -> List[tuple]:
    '''Parses onehot feature definitions from CLI.
    Items of a onehot separated by commas. Multiple different onehots separated by space.

    Args:
        onehot_feature_defs (list[str]): list of onehot feature definitions Example: ['1,2,3', '7,8']
    
    Returns:
        list(tuple): List of tuples representing one-hot encoded features. Example: [(1,2,3), (7,8)]
    '''
    return [tuple([int(i.strip()) for i in ohe_def.split(',')]) for ohe_def in onehot_feature_defs]

def _parse_ordinal_features(ordinal_feature_defs:List[str]) -> List[Tuple[int, Tuple[Union[int, float]]]]:
    '''Parses ordinal categorical feature definitions from CLI.
    Ordinal feature and values separated by comma(s). Multiple different onehots separated by space.

    Args:
        ordinal_feature_defs (list[str]): list of ordinal feature definitions Example: ['1,1.0,2.0', '7,2.0,3.0']
    
    Returns:
        list(tuple): List of tuples representing ordinal encoded features. Example: [(1, (1.0, 2.0)), (7, (2.0, 3.0))]
    '''
    defs = []
    for ord_def in ordinal_feature_defs:
        values = list(map(lambda s:s.strip(), ord_def.split(',')))
        defs.append(tuple(int(values[0]), tuple(map(float, values[1:]))))
    return defs

def _parse_allowed_misclassifications(allowed_misclassifications:List[str]) -> List[tuple]:
    '''Parses allowed misclassification definitions from CLI.
    Actual and predicted classe separated by comma; Multiple misclassifications separated by space.

    Args:
        allowed_misclassifications (list[str]): list of strings. Example: ['4,3', '3,2', '2,1']
    
    Returns:
        list(tuple): List of tuples representing allowed misclassifications. Example: [(4,3), (3,2), (2,1)]
    '''
    return [tuple([int(i.strip()) for i in amc.split(',')[0:2]]) for amc in allowed_misclassifications]

def count_decimal_places(f:float) -> int:
    '''Counts number of decimal places in float

    Args:
        f (float): The floating point number

    Returns:
        int: Number of decimal places in float
    '''
    return abs(Decimal(str(f)).as_tuple().exponent)

def ms_since_1970() -> int:
    '''Returns UNIX timestamp as milliseconds since 1970

    Returns:
        int: milliseconds since 1970
    '''
    return int(round(time.time() * 1000))

def _ms_to_human(ms:int) -> str:
    '''converts milliseconds to human-readable string

    Args:
        ms (int): number of milliseconds

    Returns:
        str: human-readable string in format "[h hours], [m minutes], [s seconds]" OR "[ms milliseconds]"
    '''
    if ms < 1000:
        return f'{ms} milliseconds'
    seconds = int((ms / 1000) % 60)
    minutes = int((ms / (1000 * 60)) % 60)
    hours = int((ms / (1000 * 60 * 60)) % 24)
    output = f'{seconds} seconds'
    output = f'{minutes} minutes, {output}' if minutes or hours else output
    output = f'{hours} hours, {output}' if hours else output
    return output

def create_logger(name:str, level:int=logging.DEBUG, to_file:bool=False, to_console:bool=True, logdir:str='logs') -> logging.Logger:
    '''Creates and configures a Logger instance

    Args:
        name (str): Name of logger
        level (int, optional): Logging level. Defaults to logging.DEBUG.
        to_file (bool, optional): Save to file? Defaults to False.
        to_console (bool, optional): Write to console?. Defaults to True.
        logdir (str, optional): Directory to log to. Defaults to 'logs'.

    Returns:
        logging.Logger: [description]
    '''
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

def _set_tf_log_level(level:int=1):
    '''sets the tensorflow log level (0=FATAL, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG)

    Args:
        level (int, optional): integer for log level. Defaults to 1 (ERROR).
    '''
    log_levels = {0: 'FATAL', 1: 'ERROR', 2: 'WARN', 3: 'INFO', 4: 'DEBUG'}
    assert level in log_levels.keys(), f'unsupported TF log level. supported:{log_levels.keys()}'
    tf.get_logger().setLevel(log_levels.get(level))

def print_heading(text:str, separator:str='-', separator_len:int=40):
    '''Prints a heading with separators

    Args:
        text (str): heading text
        separator (str, optional): Character used in separator. Default is '-'
        separator_len (int, optional): Length of separator. Default is 40.
    '''
    print(('-'*40) + f'\n{text}\n')

def create_dirpath(outpath:str):
    '''Creates any non-existent folder(s) in the outpath

    Args:
        outpath (str): Path to a file or directory
    '''
    dirpath, _ = os.path.split(outpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

def remove_softmax_activation(model_path:str, save_path:str='') -> tf.keras.Model:
    '''Prepares a classifier with softmax activation for verification by 
    removing the softmax activation function from the output layer.

    Args:
        model_path (str): Path to model
        save_path (str, optional): Path where new model is saved. Defaults to ''.

    Returns:
        tf.keras.Model: The modified tensorflow Model object
    '''
    model = tf.keras.models.load_model(model_path)
    weights = model.get_weights()
    model.pop()
    model.add(tf.keras.layers.Dense(weights[-1].shape[0], name='dense_output'))
    model.set_weights(weights)
    if save_path:
        tf.saved_model.save(model, save_path)
    return model

def softargmax(y:np.array) -> np.array:
    '''Applies softmax & argmax to emulate a softmax output layer

    Args:
        y (np.array): Logits layer output

    Returns:
        np.array: onehot encoded prediction (e.g. [0,0,1,0])
    '''
    out = np.zeros(y.shape[0], dtype=int)
    out[np.argmax(softmax(y))] = 1
    return out

def get_file_extension(filepath:str) -> str:
    '''Gets the extension from a filepath.

    Args:
        filepath (str): Path to the file

    Returns:
        str: The file's extension (e.g. '.txt')
    '''
    return Path(filepath).suffix

# # TODO: Integrate TOTUtils with TOTNet or move to new module
# class TOTUtils:
#     _features = ('FixationDuration', 'FixationSeq', 'FixationStart', 'FixationX', 
#                  'FixationY', 'GazeDirectionLeftZ', 'GazeDirectionRightZ', 'PupilLeft', 
#                  'PupilRight', 'InterpolatedGazeX', 'InterpolatedGazeY', 'AutoThrottle', 
#                  'AutoWheel', 'CurrentThrottle', 'CurrentWheel', 'Distance3D', 'MPH', 
#                  'ManualBrake', 'ManualThrottle', 'ManualWheel', 'RangeW', 'RightLaneDist', 
#                  'RightLaneType', 'LeftLaneDist', 'LeftLaneType')
#     _categories = ('fast', 'med_fast', 'med', 'med_slow', 'slow')

#     @staticmethod
#     def get_feature_names():
#         return list(TOTUtils._features)

#     @staticmethod
#     def get_feature_name(input_x):
#         return TOTUtils.get_feature_names()[input_x]
    
#     @staticmethod
#     def get_feature_index(feature_name):
#         return TOTUtils.get_feature_names().index(feature_name)
    
#     @staticmethod
#     def get_category_names():
#         return list(TOTUtils._categories)

#     @staticmethod
#     def get_category_name(y_index):
#         return TOTUtils.get_category_names()[y_index]
    
#     @staticmethod
#     def get_category_index(label):
#         return TOTUtils.get_category_names().index(label)
    
#     @staticmethod
#     def get_majority_category_index(output_sample):
#         return output_sample.index(max(output_sample))

#     @staticmethod
#     def get_scaled_value(scikit_scaler,  col_idx, actual_value):
#         '''
#         returns the scaled value of a single column
#         '''
#         dummy = [[0] * len(scikit_scaler.scale_)]
#         dummy[0][col_idx] = actual_value
#         return scikit_scaler.transform(dummy)[0][col_idx]

#     @staticmethod
#     def get_actual_value(scikit_scaler, col_idx, scaled_value):
#         '''
#         returns the actual value of a single column
#         '''
#         dummy = [[0] * len(scikit_scaler.scale_)]
#         dummy[0][col_idx] = scaled_value
#         return scikit_scaler.inverse_transform(dummy)[0][col_idx] 

#     @staticmethod
#     def get_scaled_values(scikit_scaler, actual_values):
#         '''returns the scaled values
#         '''
#         assert(len(scikit_scaler.scale_) == len(actual_values))
#         return scikit_scaler.transform(actual_values)[0]

#     @staticmethod
#     def get_actual_values(scikit_scaler, scaled_values):
#         '''
#         returns the scaled value of a single column from
#         '''
#         assert(len(scikit_scaler.scale_) == len(scaled_values))
#         return scikit_scaler.inverse_transform(scaled_values)[0]

#     @staticmethod
#     def load_csv(csv_path, frac=1):
#         '''
#         loads a csv, and returns two dataframes (inputs and outputs)

#         @param csv_path (string): path to csv file
#         @frac (float): fraction of dataset to return (default=1.0)
#         '''
#         df = pd.read_csv(csv_path, index_col=0)
#         df = df.sample(frac=frac)
#         output_cols = [f'TOT_{c}' for c in TOTUtils._categories]
#         return df.drop(output_cols, axis=1), df[output_cols]

#     @staticmethod
#     def load_samples(csv_path, frac=1):
#         '''
#         loads a csv, and converts to a list of samples [([x0-xN], [y0..yM]), ([x0..xM], [y0..yM])]

#         @param csv_path (string): path to csv file
#         @frac (float): fraction of dataset to return (default=1.0)
#         '''
#         inputs_df, outputs_df = TOTUtils.load_csv(csv_path, frac=frac)
#         return [(list(inputs_df.iloc[i]), list(outputs_df.iloc[i])) for i in range(inputs_df.shape[0])]

#     @staticmethod
#     def group_samples(samples):
#         '''
#         groups samples by category. returns a list containing samples for each category.
#         '''
#         n_outputs = len(samples[0][1])
#         groups = [[] for i in range(n_outputs)]
#         for sample in samples:
#             _, outputs = sample
#             y = outputs.index(max(outputs))
#             groups[y].append(sample)
#         return groups

#     @staticmethod
#     def filter_samples(samples, nnet_path, incorrect=False):
#         '''
#         evaluates samples and filters based on correct or incorrect prediction

#         @param samples (list): list of tuples containing inputs and outputs
#         @param nnet_path (string): path to nnet model
#         @incorrect (bool): if true, returns the incorrect predictions (default False)
#         '''
#         filtered = []
#         net = TOTNet(nnet_path)
#         for sample in samples:
#             inputs, outputs = sample
#             y = outputs.index(max(outputs))
#             is_correct = net.check_prediction(inputs, y)
#             if (not incorrect and is_correct) or (incorrect and not is_correct):
#                 filtered.append(sample)
#         return filtered
    
#     @staticmethod
#     def save_samples_to_csv(samples, outdir):
#         '''
#         saves samples to a single csv file

#         @param samples (list): list of input samples (tuples contianing inputs & outputs)
#         @param output_file (string): output file path
#         '''
#         if not os.path.exists(outdir):
#             os.makedirs(outdir, mode=0o755)
#         outfile = os.path.join(outdir, 'samples.csv')
#         df = pd.concat([
#             pd.DataFrame([s[0] for s in samples], columns=TOTUtils._features),
#             pd.DataFrame([s[1] for s in samples], columns=[f'TOT_{c}' for c in TOTUtils._categories])
#             ], axis=1)
#         df.to_csv(outfile)
#         return outfile
    
#     @staticmethod
#     def build_adversarial_samples(self, target_samples, outdir):
#         pass
