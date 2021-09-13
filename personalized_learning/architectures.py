import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model, Sequential

def arch_safescad(input_shape=None,
          initializers='he_normal',
          classes=5):

    if input_shape is None:
        raise ValueError(
            "The shape of the input layer must be not be `None`."
        )

    '''
        Model architecture:
        Input Neurons: 25(Number of features in each data point)
        Hidden Layers: Dense layers with 50-100-35-11
        Output Neurons: 5 (Number of classes)
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape)))
    model.add(Dense(50, activation='relu', kernel_initializer=initializers))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dense(11, activation='relu'))
    model.add(Dense(classes, activation='softmax'))  # logits layer
    return model

def arch_safescad_transfer(base_model=None):
    if base_model is None:
        raise ValueError(
            "The base model must be not be `None`."
        )

    # Freeze the weights learnt by the baseline model
    base_model.trainable = False

    # Consider upto the penultimate layer of the network
    base_model.pop()
    base_model.pop()

    # **Inserting additional layers in the baseline**
    # 1. Create a dense layer which will take output of base_model as its input.
    # 2. Add the final classification layer as per the baseline.

    # **A wider and deeper architecture for transfer learning**
    fc = Dense(50, activation='relu')(base_model.output)
    fc2 = Dense(25, activation='relu')(fc)
    fc3 = Dense(5, activation='softmax')(fc2)

    model = Model(inputs=base_model.input, outputs=fc3)
    return model
