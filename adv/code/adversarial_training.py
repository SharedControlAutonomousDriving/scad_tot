
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier, KerasClassifier
import os 
import logging 



# Configure a logger to capture ART outputs [printed in console,the level of detail=INFO]
logger = logging.getLogger()
logger.setLevel(logging.INFO)
LOG_DIR = '/content/drive/MyDrive/SCAD/logs'
try:
    os.mkdir("/content/drive/MyDrive/SCAD/logs")
except:
    print("Folder may already exist.")
    
    
fileHandler = logging.FileHandler("{0}/{1}.log".format(LOG_DIR, "adv_training"))
logger.addHandler(fileHandler)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  

def load_data():
    train = pd.read_csv('/content/drive/MyDrive/SCAD/data/v3.2.2_train.csv')
    train = train.drop(columns=['Unnamed: 0'])
    test = pd.read_csv('/content/drive/MyDrive/SCAD/data/v3.2.2_test.csv')
    test = test.drop(columns=['Unnamed: 0'])
    return train, test
    
def train_step(model, data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

def save_adv_model(adv_model, exp):
    # path = os.path.join(os.getcwd(),'models/')
    path = '/content/drive/MyDrive/SCAD/models/'

    try:
        os.mkdir(path)
    except:
        print("Folder may already exist.")

    filename = path + 'adv_model{}.h5'.format(exp)
    adv_model.model.save(filename)


def save_samples(samples, filename, exp):
    # path = os.path.join(os.getcwd(),'adv_data/')
    path = '/content/drive/MyDrive/SCAD/adv_data/'
    
    try:
        os.mkdir(path)
    except:
        print("Folder may already exist.")

    data_path = path + '{}_{}.csv'.format(filename, exp)
    adv_data = pd.DataFrame(samples)
    adv_data.to_csv(data_path, index=False)





if __name__ == "__main__":
    exp = 0
    eps = 0.0137
    
    logger.info("="*50)
    logger.info("Experiment : {}".format(exp))
    logger.info("Epsilon:{}".format(eps))
    logger.info("="*50)
 
    
    print("Loading model...\n")
    new_model = tf.keras.models.load_model("/content/drive/MyDrive/SCAD/network/models/latest/model.h5")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


    print("Loading data...\n")
    train, test = load_data()



    print("Setting up train and test data...\n")
    train_data, train_labels = np.split(train.to_numpy(), [25], axis=1)
    test_data, test_labels = np.split(test.to_numpy(), [25], axis=1)
    non_encoded_test_labels = tf.argmax(test_labels, axis=1)
    non_encoded_train_labels = tf.argmax(train_labels,axis=1)
    
    print("Evaluating clean samples on clean model...\n")
    _, orig_acc = new_model.evaluate(test_data, test_labels, verbose=1)
    logger.info("Classifier with original training")
    logger.info("Accuracy on clean test samples: %.2f%%", (orig_acc * 100))
    logger.info("="*50)
    

    print("Creating classifier...\n")
    adv_classifier = TensorFlowV2Classifier(
        model=new_model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=5,
        input_shape=(1,25),
        clip_values=(0, 1),
    )
    
    
    print("Creating adversarial attack object...\n")
    fgsm = FastGradientMethod(adv_classifier, 
                              norm=np.inf, 
                              eps=eps, 
                              eps_step=0.001, 
                              targeted=False, 
                              batch_size=2048, 
                              num_random_init=27)


    print("Generating adversarial samples...\n")
    logger.info("Craft attack on training examples")
    x_train_adv = fgsm.generate(train_data)
    save_samples(x_train_adv, 'adv_train', exp)

    logger.info("Craft attack test examples")
    x_test_adv = fgsm.generate(test_data)
    save_samples(x_test_adv, 'adv_test', exp)
    

    print("Evaluating adversarial samples on clean model...\n")
    preds = np.argmax(adv_classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == non_encoded_test_labels) / non_encoded_test_labels.shape[0]
    logger.info("Classifier before adversarial training")
    logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))
    logger.info("="*50)

    
    
    print("Augmenting original data with adversarial samples...\n")
    aug_train_data = np.append(train_data, x_train_adv, axis=0)
    aug_train_labels = np.append(train_labels, train_labels, axis=0)

    
    print("Retraining model on new dataset...\n")
    new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    adv_classifier.fit(aug_train_data, aug_train_labels, nb_epochs=10, batch_size=2048)
    save_adv_model(adv_classifier, exp)
    print("Saved model...\n")
    
    print("Evaluate adversarially trained model on adversarial test samples...\n")
    preds_adv = np.argmax(adv_classifier.predict(x_test_adv), axis=1)
    acc_adv = np.sum(preds_adv == non_encoded_test_labels) / non_encoded_test_labels.shape[0]
    logger.info("Classifier with adversarial training")
    logger.info("Accuracy on adversarial test samples: %.2f%%", (acc_adv * 100))
    logger.info("="*50)

    print("Evaluate adversarially trained model on clean test samples...\n")
    new_preds_clean = np.argmax(adv_classifier.predict(test_data), axis=1)
    new_acc_clean = np.sum(new_preds_clean == non_encoded_test_labels) / non_encoded_test_labels.shape[0]
    logger.info("Classifier with adversarial training")
    logger.info("Accuracy on clean test samples: %.2f%%", (new_acc_clean * 100))
    logger.info("="*50)
    

    
    
    
    

    







    
