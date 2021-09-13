import os, logging, random, time, math, zipfile
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Union
from pathlib import Path
from decimal import Decimal
from scipy.special import softmax
from tensorflow.keras.models import load_model

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

def download_and_unzip(src_url, out_dir='./', zip_file='dl.zip', remove_zip=True):
    print(f'Downloading {src_url} to {zip_file}')
    os.system("wget " + src_url + " -O " + zip_file + " -q --show-progress")
    print(f'Download complete. Unzipping {zip_file}')
    z = zipfile.ZipFile(zip_file, 'r')
    z.extractall(out_dir)
    print(f'Unzipped to {out_dir}. Cleaning up...')
    z.close()
    if remove_zip: os.remove(zip_file)

def update_path_vars(paths=[]):
    python_path = os.environ.get('PYTHONPATH') or ''
    jupyter_path = os.environ.get('JUPYTER_PATH') or ''
    for path in paths:
        if not path in python_path:
            python_path += f':{path}'
        if not path in jupyter_path:
            jupyter_path += f':{path}'
    os.environ['PYTHONPATH'] = python_path
    os.environ['JUPYTER_PATH'] = jupyter_path

def install_nnet_tools(nnet_tools_path):
    nnet_tools_url = 'https://github.com/sisl/NNet/archive/master.zip'
    download_and_unzip(nnet_tools_url)
    os.system("mv ./NNet-master " + nnet_tools_path)

def setup_nnet_tools(nnet_tools_path):
    # install nnet tools if not already installed.
    if not os.path.exists(nnet_tools_path):
        install_nnet_tools(nnet_tools_path)
    # add folder to PYTHONPATH & JUPYTER_PATH
    update_path_vars([nnet_tools_path])

def compute_nnet_params(model_file, enc_inputs):
    # compute sdev, mins, and maxs for inputs
    input_sdev = np.std(enc_inputs, axis=0)
    input_mins = np.amin(enc_inputs, axis=0)
    input_maxs = np.amax(enc_inputs, axis=0)

    # extend input maxs and mins by std dev
    input_mins -= input_sdev
    input_maxs += input_sdev

    # maraboupy only supports normalization (not standardization)
    # use mean=0, and range=1 to neutralize maraboupy normalization
    means = np.zeros(enc_inputs.shape[1]+1, dtype=int)
    ranges = np.ones(enc_inputs.shape[1]+1, dtype=int)

    # extract weights and biases from model
    model = load_model(model_file)
    model_params = model.get_weights()
    weights = [w.T for w in model_params[0:len(model_params):2]]
    biases  = model_params[1:len(model_params):2]

    return (weights, biases, input_mins, input_maxs, means, ranges)

def save_nnet(weights, biases, input_mins, input_maxs, means, ranges, output_path):
    # write model in nnet format.
    from NNet.utils.writeNNet import writeNNet
    writeNNet(weights, biases, input_mins, input_maxs, means, ranges, output_path)