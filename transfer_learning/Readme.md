##### Content:
This folder contains the code and the checkpoints for the transfer learning experiments.

##### Aim:
The idea is to obtain a deep learning model which is personalized for a specific driver.
This stems from the fact that there could be limited amount of data available for a new driver,
so we'd benefit a lot from leveraging the initial weights learnt by the baseline model and then
perform transfer learning for the new data.

##### Data:
The data is collected from simulation experiments and is inclusive of data from 15 different individuals,
each performing 3 trials of the experiments. The dataframe has 5909111 rows and each row has 28 features.
Out of the 28 features, 25 are retained to train the baseline model. The train test split is 80-20.

##### Methodology and Baseline:
The baseline model is trained on driver data excluding the one originating from the specified driver for whom
the personalization is happening. Baseline architecture is a 4 layer deep dense neural network with 25 input neurons
and 5 output neurons. The hidden layers contain 50,100,35,11 neurons respectively in each succeeding layer. The hyperparameters for
training the baseline are as follows:

1. Hidden layer activation function- ReLU.
2. Output layer activation function- Softmax.
3. Loss function- Categorical CrossEntropy
4. Optimizer- SGD
5. Batch Size- 512
6. Number of epochs- 200
7. Initial learning rate- 0.01
8. Learning Rate scheduler- ReduceLROnPlateau with patience of 15 epochs and a reduction factor of 0.2

The above configuration takes close to 30 seconds per epoch to train on the data consisting of 14 drivers.

##### Transfer learning architecture:
For the purpose of transfer learning the first 3 hidden layers are retained and the rest are chopped off.
Two transfer learning layers, with 50 and 35 neurons respectively are added before the output layer. These new
layers are then trained solely on the new driver's data.
The hyperparameters follow the baseline architecture itself. The number of epochs is reduced to 80 with 4 seconds
to train per epoch.

Run the LR_scheduler notebook to infer the results.
