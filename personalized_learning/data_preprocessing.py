import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from functools import reduce
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


##############################################################
# SafeSCAD specific functions
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

def get_safescad_data(dataset_file, data_dir, new_driver_ids):
    print('Getting SafeScad dataset...')
    if new_driver_ids is None and dataset_file is not None:
        raise ValueError("new_driver_ids must not be `None` if dataset_file is not None.")
    n_categories = len(create_tot_categories()[1])
    if dataset_file is not None:
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

        # make a copy the raw data
        full_df = raw_df

        # compute 'TOT' categories
        tot_bins, tot_labels = create_tot_categories(full_df.ReactionTime)
        # print the TOT categories
        print('TOT Categories')
        print('\n'.join(
            ['%s: %9.2f, %7.2f' % (tot_labels[i].rjust(8), tot_bins[i], tot_bins[i + 1]) for i in
             range(n_categories)]))

        full_df.RightLaneType = full_df.RightLaneType.astype(int)
        full_df.LeftLaneType = full_df.LeftLaneType.astype(int)

        # add the class to the dataframe
        full_df['TOT'] = pd.cut(full_df.ReactionTime, bins=tot_bins, labels=tot_labels).astype(object)

        # prepare encoders
        scaler = prepare_inputs(full_df.drop(['Name', 'ReactionTime', 'TOT'], axis=1))
        onehot = prepare_target(full_df.TOT, categories=[tot_labels])

        # Personalization dataset for new driver
        new_driver = new_driver_ids.split(";")

        # Selecting new driver's data from full dataset.
        only_new_df = full_df.loc[full_df['Name'].isin(new_driver)]
        ids_new = full_df['Name'].isin(new_driver)

        # Dropping data concerning driver from main training dataset
        base_df = full_df.copy()
        base_df.drop(base_df[ids_new].index, inplace=True)

        print('Full dataframe without new driver data', base_df.shape)
        print('New driver dataframe', only_new_df.shape)

        # split features and targets
        y_full = full_df.TOT
        X_full = full_df.drop(['Name', 'ReactionTime', 'TOT'], axis=1)
        y_base = base_df.TOT
        X_base = base_df.drop(['Name', 'ReactionTime', 'TOT'], axis=1)
        y_new = only_new_df.TOT
        X_new = only_new_df.drop(['Name', 'ReactionTime', 'TOT'], axis=1)

        # display the feature names
        feature_names = list(X_full.columns)
        print('Feature Names', feature_names)

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
        print('TOT Value Counts in full data before upsampling\n', y_full_train.value_counts())
        print('TOT Value Counts in base data before upsampling\n', y_base_train.value_counts())
        print('TOT Value Counts in new data before upsampling\n', y_new_train.value_counts())
        X_full_train, y_full_train = upsample_minority_TOTs(X_full_train, y_full_train, tot_labels)
        X_base_train, y_base_train = upsample_minority_TOTs(X_base_train, y_base_train, tot_labels)
        X_new_train, y_new_train = upsample_minority_TOTs(X_new_train, y_new_train, tot_labels)
        print('TOT Value Counts in full data after upsampling\n', y_full_train.value_counts())
        print('TOT Value Counts in base data after upsampling\n', y_base_train.value_counts())
        print('TOT Value Counts in new data after upsampling\n', y_new_train.value_counts())


        # scale the inputs
        X_full_train_enc = scaler.transform(X_full_train.values)
        X_full_test_enc = scaler.transform(X_full_test.values)
        X_base_train_enc = scaler.transform(X_base_train.values)
        X_base_test_enc = scaler.transform(X_base_test.values)
        X_new_train_enc = scaler.transform(X_new_train.values)
        X_new_test_enc = scaler.transform(X_new_test.values)

        # categorize outputs
        y_full_train_enc = onehot.transform(y_full_train.to_numpy().reshape(-1,1)).toarray()
        y_full_test_enc = onehot.transform(y_full_test.to_numpy().reshape(-1,1)).toarray()
        y_base_train_enc = onehot.transform(y_base_train.to_numpy().reshape(-1,1)).toarray()
        y_base_test_enc = onehot.transform(y_base_test.to_numpy().reshape(-1,1)).toarray()
        y_new_train_enc = onehot.transform(y_new_train.to_numpy().reshape(-1,1)).toarray()
        y_new_test_enc = onehot.transform(y_new_test.to_numpy().reshape(-1, 1)).toarray()

        print("Saving train and test data for the base and new scenarios ...")
        # save_data(X_full_train_enc, y_full_train_enc, feature_names, onehot, 'TOT', 'train_full', data_dir)
        # save_data(X_full_test_enc, y_full_test_enc, feature_names, onehot, 'TOT', 'test_full', data_dir)
        save_data(X_base_train_enc, y_base_train_enc, feature_names, onehot, 'TOT', 'train_base', data_dir)
        save_data(X_base_test_enc, y_base_test_enc, feature_names, onehot, 'TOT', 'test_base', data_dir)
        save_data(X_new_train_enc, y_new_train_enc, feature_names, onehot, 'TOT', 'train_new', data_dir)
        save_data(X_new_test_enc, y_new_test_enc, feature_names, onehot, 'TOT', 'test_new', data_dir)
        save_encoders(scaler, onehot, data_dir)

        return (#X_full_train_enc, y_full_train_enc, X_full_test_enc, y_full_test_enc,
                X_base_train_enc, y_base_train_enc, X_base_test_enc, y_base_test_enc,
                X_new_train_enc, y_new_train_enc, X_new_test_enc, y_new_test_enc)

    else:
        # full_train_data = pd.read_csv(f'{data_dir}/train_full.csv')
        # full_test_data = pd.read_csv(f'{data_dir}/test_full.csv')
        base_train_data = pd.read_csv(f'{data_dir}/train_base.csv')
        base_test_data = pd.read_csv(f'{data_dir}/test_base.csv')
        new_train_data = pd.read_csv(f'{data_dir}/train_new.csv')
        new_test_data = pd.read_csv(f'{data_dir}/test_new.csv')

        # y_full_train_enc = full_train_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        # y_full_test_enc = full_test_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        y_base_train_enc = base_train_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        y_base_test_enc = base_test_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        y_new_train_enc = new_train_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]
        y_new_test_enc = new_test_data[['TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow']]

        # X_full_train_enc = full_train_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        # X_full_test_enc = full_test_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        X_base_train_enc = base_train_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        X_base_test_enc = base_test_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        X_new_train_enc = new_train_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)
        X_new_test_enc = new_test_data.drop(['Unnamed: 0', 'TOT_fast', 'TOT_med_fast', 'TOT_med', 'TOT_med_slow', 'TOT_slow'], axis=1)

        # display the feature names
        feature_names = list(X_base_train_enc.columns)
        print('Feature Names', feature_names)

        return (#X_full_train_enc.values, y_full_train_enc.values, X_full_test_enc.values, y_full_test_enc.values,
                X_base_train_enc.values, y_base_train_enc.values, X_base_test_enc.values, y_base_test_enc.values,
                X_new_train_enc.values, y_new_train_enc.values, X_new_test_enc.values, y_new_test_enc.values)
##############################################################
##############################################################
# Generic functions
def prepare_inputs(X):
    # scales inputs using "standard scaler", and returns 2D numpy array
    scaler = StandardScaler().fit(X)
    return scaler

def prepare_target(y, categories):
    # convert target to categorical, and returns 2D numpy array
    y = y.to_numpy().reshape(-1,1)
    onehot = OneHotEncoder(categories=categories)
    onehot.fit(y)
    return onehot

def save_encoders(scaler, onehot, output_dir):
    if scaler is not None:
        pkl.dump(scaler, open(f'{output_dir}/scaler.pkl', 'wb'))
    if onehot is not None:
        pkl.dump(onehot, open(f'{output_dir}/onehot.pkl', 'wb'))

def save_data(X_enc, y_enc, features, onehot, onehot_name, fname, data_dir='../data'):
    labels = onehot.get_feature_names(input_features=[onehot_name])
    df = pd.concat([pd.DataFrame(X_enc, columns=features),
                          pd.DataFrame(y_enc, columns=labels)],
                        axis=1).astype({k:int for k in labels})
    data_csv = f'{data_dir}/{fname}.csv'
    df.to_csv(data_csv)
    print(f'wrote data to {data_csv}')

def get_data(experiment, dataset_file, data_dir, new_driver_ids=None):
    print("Loading and preprocessing data ...")
    get_data = globals()[f'get_{experiment}_data']
    return get_data(dataset_file, data_dir, new_driver_ids)
