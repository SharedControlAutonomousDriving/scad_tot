import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
import os 
import logging

# Configure a logger to capture ART outputs [printed in console,the level of detail=INFO]
logger = logging.getLogger()
logger.setLevel(logging.INFO)
LOG_DIR = '/content/drive/MyDrive/SCAD/logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
fileHandler = logging.FileHandler("{0}/{1}.log".format(LOG_DIR, "adv_training"))
logger.addHandler(fileHandler)

handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

    

def load_data():
    train = pd.read_csv('/content/drive/MyDrive/SCAD/data/v3.2.2_train.csv')
    train = train.drop(columns=['Unnamed: 0'])
    test = pd.read_csv('/content/drive/MyDrive/SCAD/data/v3.2.2_test.csv')
    test = test.drop(columns=['Unnamed: 0'])
    return train, test
    
def train_step(model, data, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    



if __name__ == "__main__":
    #Loading model
    new_model = tf.keras.models.load_model("/content/drive/MyDrive/SCAD/network/models/latest/model.h5")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    #Loading data
    train, test = load_data()

    #Convert Pandas data to numpy arrays
    train_data, train_labels = np.split(train.to_numpy(), [25], axis=1)
    test_data, test_labels = np.split(test.to_numpy(), [25], axis=1)
    non_encoded_test_labels = tf.argmax(test_labels, axis=1)
    non_encoded_train_labels = tf.argmax(train_labels,axis=1)

    #Create adversarial classifier
    adv_classifier = TensorFlowV2Classifier(
        model=new_model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=5,
        input_shape=(1,25),
        clip_values=(0, 1),
    )
    
    #Create adversarial object using ART 
    fgsm = FastGradientMethod(adv_classifier, norm=np.inf, eps=0.01, eps_step=0.001, targeted=False, batch_size=128, num_random_init=27)
    
    logger.info("Craft attack on training examples")
    x_train_adv = fgsm.generate(train_data)
    
    logger.info("Craft attack test examples")
    x_test_adv = fgsm.generate(test_data)
    
    #Evaluate adversarial samples on clean model.
    preds = np.argmax(adv_classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == non_encoded_test_labels) / non_encoded_test_labels.shape[0]
    logger.info("Classifier before adversarial training")
    logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))
    
    #Select 10000 adversarial samples to retrain the model with
    idx = np.random.randint(0, len(x_train_adv), 10000)
    adv_samples = np.array([x_train_adv[i] for i in idx])
    adv_sample_labels = np.array([train_labels[i] for i in idx])
    non_encoded_adv_labels = np.array([non_encoded_train_labels[i] for i in idx])
    
    # Data augmentation: expand the training set with the adversarial samples
    aug_train_data = np.append(train_data, adv_samples, axis=0)
    aug_train_labels = np.append(train_data_labels, adv_sample_labels, axis=0)
    
    # Retrain the model on the extended dataset
    new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    adv_classifier.fit(aug_train_data, aug_train_labels, nb_epochs=10, batch_size=128)

    
    # Evaluate adversarially trained classifier on the test data
    preds = np.argmax(adv_classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == non_encoded_test_labels) / non_encoded_test_labels.shape[0]
    logger.info("Classifier with adversarial training")
    logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))
    
    
    
    
    

    







    
