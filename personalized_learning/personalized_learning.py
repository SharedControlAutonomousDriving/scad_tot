import glob, math, os, re, sys, zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import Model
from scriptify import scriptify
from data_preprocessing import get_data
from utils import setup_nnet_tools, compute_nnet_params, save_nnet
import architectures

if __name__ == '__main__':

    @scriptify
    def script(experiment='safescad',
               epochs_base=200,
               epochs_new=50,
               batch_size=512,
               dataset_file=None,
               conf_name='default',
               new_driver_ids='013_M1;013_M2;013_M3',
               gpu = 0):

        # select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        # basic setup and install additional dependencies
        # some global variables and general settings
        model_dir = f'./models/{experiment}/{conf_name}'
        data_dir = f'./data/{experiment}/{conf_name}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        pd.options.display.float_format = '{:.2f}'.format
        nnet_tools_path = os.path.abspath('NNet')

        # setup nnet tools (for converting model to Stanford's nnet format)
        setup_nnet_tools(nnet_tools_path)

        # Load and Preprocess Dataset
        X_base_train_enc, y_base_train_enc, X_base_test_enc, y_base_test_enc, \
        X_new_train_enc, y_new_train_enc, X_new_test_enc, y_new_test_enc \
            = get_data(experiment, dataset_file, data_dir, new_driver_ids)

        ## build & Train NN
        # training callbacks
        es_cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
        # mc_file = 'model-best-{epoch:02d}-{val_loss:.2f}.h5'
        # mc_cb = ModelCheckpoint(mc_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # tb_cb = TensorBoard(log_dir=tensorboard_logs, histogram_freq=1, write_graph=True, write_images=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=0.0001)

        n_categories = y_base_train_enc.shape[1]
        arch = getattr(architectures, f'arch_{experiment}')
        model = arch((X_base_train_enc.shape[1],), classes=n_categories)

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.summary()

        # fit the keras model on the base dataset
        history = model.fit(X_base_train_enc, y_base_train_enc,
                            validation_split=0.10,
                            epochs=epochs_base,
                            batch_size=batch_size,
                            callbacks=[es_cb, reduce_lr])

        # save model in tf and h5 formats
        tf_model_path = f'{model_dir}/model_base.tf'
        h5_model_path = f'{model_dir}/model_base.h5'
        model.save(tf_model_path, save_format='tf')
        model.save(h5_model_path, save_format='h5')

        # extract params for nnet format
        nnet_params = compute_nnet_params(tf_model_path, np.concatenate((X_base_train_enc,X_base_test_enc)))
        weights, biases, input_mins, input_maxs, means, ranges = nnet_params
        # write the model to nnet file.
        nnet_path = os.path.join(model_dir, f'model_base.nnet')
        save_nnet(weights, biases, input_mins, input_maxs, means, ranges, nnet_path)

        """ Evaluate base model"""
        print('Evaluating base model ...')

        # model being tested on data without new driver*
        _, train_acc = model.evaluate(X_base_train_enc, y_base_train_enc, verbose=2)
        _, test_acc = model.evaluate(X_base_test_enc, y_base_test_enc, verbose=1)
        print('Accuracy of the base model on base data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))

        # model being tested on new driver's data**
        _, train_acc = model.evaluate(X_new_train_enc, y_new_train_enc, verbose=2)
        _, test_acc = model.evaluate(X_new_test_enc, y_new_test_enc, verbose=1)
        print('Accuracy of the base model on new driver\'s data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))


        # fine-tuning the model on new driver's data

        history = model.fit(X_new_train_enc, y_new_train_enc,
                            validation_split=0.10,
                            epochs=epochs_new,
                            batch_size=512,
                            callbacks=[es_cb, reduce_lr])

        # save model in tf and h5 formats
        tf_model_path = f'{model_dir}/model_new.tf'
        h5_model_path = f'{model_dir}/model_new.h5'
        model.save(tf_model_path, save_format='tf')
        model.save(h5_model_path, save_format='h5')

        # extract params for nnet format
        nnet_params = compute_nnet_params(tf_model_path, np.concatenate((X_base_train_enc, X_base_test_enc)))
        weights, biases, input_mins, input_maxs, means, ranges = nnet_params
        # write the model to nnet file.
        nnet_path = os.path.join(model_dir, f'model_new.nnet')
        save_nnet(weights, biases, input_mins, input_maxs, means, ranges, nnet_path)

        # evaluate transferred model
        print('Evaluating transferred model ...')

        # evaluate the model on new driver's data
        _, train_acc = model.evaluate(X_new_train_enc, y_new_train_enc, verbose=2)
        _, test_acc = model.evaluate(X_new_test_enc, y_new_test_enc, verbose=1)
        print('Accuracy of the transferred model on new driver\'s data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))

        # does transferred model generalize at least a little?
        # evaluate on base data
        _, train_acc = model.evaluate(X_base_train_enc, y_base_train_enc, verbose=2)
        _, test_acc = model.evaluate(X_base_test_enc, y_base_test_enc, verbose=1)
        print('Accuracy of the transferred model on the base data: '
              '' + '1) Train: %.3f, 2) Test: %.3f' % (train_acc, test_acc))