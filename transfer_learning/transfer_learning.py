import glob, math, os, re, sys, zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
from functools import reduce
from itertools import cycle
from datetime import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import Model
from scriptify import scriptify

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

# 5 class using mean & sdev
def create_tot_categories(rt_column=None):
    bins = [0, 0, 0, 0, 0, 0]
    labels = np.array(['fast', 'med_fast', 'med', 'med_slow', 'slow'], dtype=object)

    if rt_column is None:
        return (bins,labels)

    rt_mean = round(rt_column.mean())
    rt_sdev = round(rt_column.std())
    bound_1 = rt_mean - rt_sdev
    bound_2 = rt_mean - rt_sdev // 2
    bound_3 = rt_mean + rt_sdev // 2
    bound_4 = rt_mean + rt_sdev
    bins = [float('-inf'), bound_1, bound_2, bound_3, bound_4, float('inf')]
    return (bins, labels)

def upsample_minority_TOTs(X_train, y_train, tot_labels, random_state=27):
    # contat the training data together.
    X = pd.concat([X_train, y_train], axis=1)
    # separate majority and minority classes
    buckets = {l: X[X.TOT == l] for l in tot_labels}
    maj_label, majority = reduce(lambda a,b: b if b[1].shape[0] > a[1].shape[0] else a, buckets.items())
    minorities = {k:v for k,v in buckets.items() if k != maj_label}
    # upsample the minority classes
    for k,v in minorities.items():
        buckets[k] = resample(v, replace=True, n_samples=majority.shape[0], random_state=random_state)
    upsampled = pd.concat(buckets.values()).sample(frac=1)
    # split the upsampled data into X and y
    y_train = upsampled['TOT']
    X_train = upsampled.drop('TOT', axis=1)
    return X_train, y_train

def prepare_inputs(X_train, X_test):
    # scales inputs using "standard scaler", and returns 2D numpy array
    scaler = StandardScaler().fit(pd.concat([X_train, X_test]))
    X_train = scaler.transform(X_train.values)
    X_test = scaler.transform(X_test.values)
    return X_train, X_test, scaler

def prepare_target(y_train, y_test, categories):
    # convert target to categorical, and returns 2D numpy array
    y_train = y_train.to_numpy().reshape(-1,1)
    y_test = y_test.to_numpy().reshape(-1,1)
    onehot = OneHotEncoder(categories=categories)
    onehot.fit(np.concatenate([y_train, y_test]))
    y_train = onehot.transform(y_train).toarray()
    y_test = onehot.transform(y_test).toarray()
    return y_train, y_test, onehot

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

def save_encoders(scaler, onehot, output_dir, mode):
    pkl.dump(scaler, open(f'{output_dir}/scaler{mode}.pkl', 'wb'))
    pkl.dump(onehot, open(f'{output_dir}/onehot{mode}.pkl', 'wb'))

def save_data(X_train_enc, X_test_enc, y_train_enc, y_test_enc, features, onehot, data_dir='../data',
              mode=''):
    tot_labels = onehot.get_feature_names(input_features=['TOT'])
    train_df = pd.concat([pd.DataFrame(X_train_enc, columns=features),
                          pd.DataFrame(y_train_enc, columns=tot_labels)],
                        axis=1).astype({k:int for k in tot_labels})
    test_df = pd.concat([pd.DataFrame(X_test_enc, columns=features),
                         pd.DataFrame(y_test_enc, columns=tot_labels)],
                        axis=1).astype({k:int for k in tot_labels})
    train_csv, test_csv = f'{data_dir}/train{mode}.csv', f'{data_dir}/test{mode}.csv'
    train_df.to_csv(train_csv)
    test_df.to_csv(test_csv)
    print(f'wrote data to {train_csv} and {test_csv}')

if __name__ == '__main__':

    @scriptify
    def script(epochs=30,
               batch_size=128,
               dataset_file='All_Features_ReactionTime.csv',
               conf_name='default',
               new_driver_ids='013_M1;013_M2;013_M3'):
        " Setup & Install """
        #Basic setup and install additional dependencies

        # Some global variables and general settings
        saved_model_dir = f'./models/{conf_name}'
        tensorboard_logs = f'./models/{conf_name}'
        load_data_dir = f'../data'
        data_dir = f'./data/{conf_name}'
        pd.options.display.float_format = '{:.2f}'.format
        nnet_tools_path = os.path.abspath('NNet')

        # setup nnet tools (for converting model to Stanford's nnet format)
        setup_nnet_tools(nnet_tools_path)

        # Load and Preprocess Dataset
        print("Loading and preprocessing data ...")
        n_categories = len(create_tot_categories()[1])

        dataset_file = f'{load_data_dir}/{dataset_file}'

        " Import Dataset """

        raw_columns = ['ID', 'Name', 'FixationDuration', 'FixationStart', 'FixationSeq',
                       'FixationX', 'FixationY', 'GazeDirectionLeftZ', 'GazeDirectionRightZ',
                       'PupilLeft', 'PupilRight', 'InterpolatedGazeX', 'InterpolatedGazeY',
                       'AutoThrottle', 'AutoWheel', 'CurrentThrottle', 'CurrentWheel',
                       'Distance3D', 'MPH', 'ManualBrake', 'ManualThrottle', 'ManualWheel',
                       'RangeW', 'RightLaneDist', 'RightLaneType', 'LeftLaneDist', 'LeftLaneType',
                       'ReactionTime']
        raw_df = pd.read_csv(dataset_file, usecols=raw_columns)
        raw_df.set_index(['ID'], inplace=True)

        # compute 'TOT' categories
        tot_bins, tot_labels = create_tot_categories(raw_df.ReactionTime)

        raw_df.RightLaneType = raw_df.RightLaneType.astype(int)
        raw_df.LeftLaneType = raw_df.LeftLaneType.astype(int)

        # add the class to the dataframe
        raw_df['TOT'] = pd.cut(raw_df.ReactionTime, bins=tot_bins, labels=tot_labels).astype(object)

        full_df = raw_df

        # Personalization dataset for new driver
        new_driver = new_driver_ids.split(";")

        # Selecting new driver's data from full dataset.
        only_new_df = full_df.loc[full_df['Name'].isin(new_driver)]
        ids_new = full_df['Name'].isin(new_driver)

        # Dropping data concerning driver from main training dataset
        base_df = full_df.copy()
        base_df.drop(base_df[ids_new].index, inplace=True)

        print('Full dataframe without new driver data', base_df.shape,
              '\n\n', 'New driver dataframe', only_new_df.shape)

        # split features and targets
        y_full = full_df.TOT
        X_full = full_df.drop(['TOT'], axis=1)
        y_base = base_df.TOT
        X_base = base_df.drop(['TOT'], axis=1)
        y_new = only_new_df.TOT
        X_new = only_new_df.drop(['TOT'], axis=1)

        # make results easier to reproduce
        random_state = 27

        # split train and test data
        X_full_train, X_full_test, y_full_train, y_full_test = \
            train_test_split(X_full, y_full, test_size=0.20, stratify=y_full, random_state=random_state)
        X_base_train, X_base_test, y_base_train, y_base_test = \
            train_test_split(X_base, y_base, test_size=0.20, stratify=y_base, random_state=random_state)
        X_new_train, X_new_test, y_new_train, y_new_test = \
            train_test_split(X_new, y_new, test_size=0.20, stratify=y_new, random_state=random_state)

        # upsample the training data
        X_full_train, y_full_train = upsample_minority_TOTs(X_full_train, y_full_train, tot_labels)
        X_base_train, y_base_train = upsample_minority_TOTs(X_base_train, y_base_train, tot_labels)
        X_new_train, y_new_train = upsample_minority_TOTs(X_new_train, y_new_train, tot_labels)

        # scale the inputs
        X_full_train_enc, X_full_test_enc, scaler_full = prepare_inputs(X_full_train, X_full_test)
        X_base_train_enc, X_base_test_enc, scaler_base = prepare_inputs(X_base_train, X_base_test)
        X_new_train_enc, X_new_test_enc, scaler_new = prepare_inputs(X_new_train, X_new_test)

        # categorize outputs
        y_full_train_enc, y_full_test_enc, onehot_full = prepare_target(y_full_train, y_full_test, categories=[tot_labels])
        y_base_train_enc, y_base_test_enc, onehot_base = prepare_target(y_base_train, y_base_test, categories=[tot_labels])
        y_new_train_enc, y_new_test_enc, onehot_new = prepare_target(y_new_train, y_new_test, categories=[tot_labels])

        print('TOT Value Counts in full dataset', y_full_train.value_counts())
        print('TOT Value Counts in base dataset', y_base_train.value_counts())
        print('TOT Value Counts in new dataset', y_new_train.value_counts())
        # print the TOT categories
        print('TOT Categories')
        print('\n'.join(
            ['%s: %9.2f, %7.2f' % (tot_labels[i].rjust(8), tot_bins[i], tot_bins[i + 1]) for i in range(n_categories)]))

        # save the column names & indexes for use during verification
        feature_names = list(X_full_train.columns)
        # display the feature names
        print('Feature Names', feature_names)

        print("Saving train and test data ...")
        save_data(X_full_train_enc, X_full_test_enc, y_full_train_enc, y_full_test_enc,
                  feature_names, onehot_full, data_dir, mode='_full')
        save_encoders(scaler_full, onehot_full, saved_model_dir, "_full")
        save_data(X_base_train_enc, X_base_test_enc, y_base_train_enc, y_base_test_enc,
                  feature_names, onehot_base, data_dir, mode='_base')
        save_encoders(scaler_base, onehot_base, saved_model_dir, "_base")
        save_data(X_new_train_enc, X_new_test_enc, y_new_train_enc, y_new_test_enc,
                  feature_names, onehot_new, data_dir, mode='_new')
        save_encoders(scaler_new, onehot_new, saved_model_dir, "_new")


        """
        ## Build & Train NN"""

        # training callbacks
        es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        # mc_file = 'model-best-{epoch:02d}-{val_loss:.2f}.h5'
        # mc_cb = ModelCheckpoint(mc_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # tb_cb = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_graph=True, write_images=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=0.0001)

        '''
        Model architecture:
        Input Neurons: 25(Number of features in each data point)
        Hidden Layers: Dense layers with 50-100-35-11
        Output Neurons: 5 (Number of classes)
        '''
        model = Sequential()
        model.add(InputLayer(input_shape=(X_full_train_enc.shape[1],)))
        model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(35, activation='relu'))
        model.add(Dense(11, activation='relu'))
        model.add(Dense(5, activation='softmax'))  # logits layer

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # fit the keras model on the base dataset
        history = model.fit(X_base_train_enc, y_base_train_enc,
                            validation_split=0.10,
                            epochs=200,
                            batch_size=512,
                            callbacks=[es_cb, reduce_lr])

        # save model in tf and h5 formats
        tf_model_path = f'{saved_model_dir}/model_base'
        h5_model_path = f'{saved_model_dir}/model_base.h5'
        model.save(tf_model_path, save_format='tf')
        model.save(h5_model_path, save_format='h5')

        # extract params for nnet format
        nnet_params = compute_nnet_params(tf_model_path, np.concatenate((X_base_train_enc,X_base_test_enc)))
        weights, biases, input_mins, input_maxs, means, ranges = nnet_params
        # write the model to nnet file.
        nnet_path = os.path.join(saved_model_dir, f'model_base.nnet')
        save_nnet(weights, biases, input_mins, input_maxs, means, ranges, nnet_path)

        """ Evaluate base model"""
        print('Evaluating base model ...')

        # load the saved base model
        base_model = load_model(tf_model_path)
        base_model.summary()

        # Model being tested on data without new driver*

        _, train_acc = base_model.evaluate(X_base_train_enc, y_base_train_enc, verbose=2)
        _, test_acc = base_model.evaluate(X_base_test_enc, y_base_test_enc, verbose=1)
        print('Accuracy of the base model on base data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))

        # **Accuracy of the base model on new driver's data**

        _, train_acc = base_model.evaluate(X_new_train_enc, y_new_train_enc, verbose=2)
        _, test_acc = base_model.evaluate(X_new_test_enc, y_new_test_enc, verbose=1)
        print('Accuracy of the base model on new driver\'s data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))

        # Freeze the weights learnt by the baseline model
        base_model.trainable = False

        # Consider upto the penultimate layer of the network
        base_model.pop()
        base_model.pop()

        # Verify the slice of the architecture
        base_model.summary()

        # **Inserting additional layers in the baseline**
        # 1. Create a dense layer which will take output of base_model as its input.
        # 2. Add the final classification layer as per the baseline.

        # **A wider and deeper architecture for transfer learning**
        fc = Dense(50, activation='relu')(base_model.output)
        fc2 = Dense(25, activation='relu')(fc)
        fc3 = Dense(5, activation='softmax')(fc2)

        model = Model(inputs=base_model.input, outputs=fc3)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.summary()

        # **Training the modified architecture**
        # The training is done only on new driver's data

        history = model.fit(X_new_train_enc, y_new_train_enc,
                            validation_split=0.10,
                            epochs=50,
                            batch_size=512,
                            callbacks=[es_cb, reduce_lr])

        # save model in tf and h5 formats
        tf_model_path = f'{saved_model_dir}/model_transferred'
        h5_model_path = f'{saved_model_dir}/model_transferred.h5'
        model.save(tf_model_path, save_format='tf')
        model.save(h5_model_path, save_format='h5')

        # extract params for nnet format
        nnet_params = compute_nnet_params(tf_model_path, np.concatenate((X_base_train_enc, X_base_test_enc)))
        weights, biases, input_mins, input_maxs, means, ranges = nnet_params
        # write the model to nnet file.
        nnet_path = os.path.join(saved_model_dir, f'model_transferred.nnet')
        save_nnet(weights, biases, input_mins, input_maxs, means, ranges, nnet_path)

        """ Evaluate transferred model"""
        print('Evaluating transferred model ...')

        # load the saved transferred model
        new_model = load_model(tf_model_path)
        new_model.summary()

        # ### Evaluate the model on new driver's data
        _, train_acc = new_model.evaluate(X_new_train_enc, y_new_train_enc, verbose=2)
        _, test_acc = new_model.evaluate(X_new_test_enc, y_new_test_enc, verbose=1)
        print('Accuracy of the transferred model on new driver\'s data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))

        # **Does transferred model generalize at least a little?**
        _, train_acc = new_model.evaluate(X_base_train_enc, y_base_train_enc, verbose=2)
        _, test_acc = new_model.evaluate(X_base_test_enc, y_base_test_enc, verbose=1)
        print('Accuracy of the transferred model on the base data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))