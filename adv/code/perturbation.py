import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from cleverhans.future.tf2.attacks import fast_gradient_method


def load_data():
    train = pd.read_csv('/content/drive/MyDrive/SCAD/data/v3.2.2_train.csv')
    train = train.drop(columns=['Unnamed: 0'])
    test = pd.read_csv('/content/drive/MyDrive/SCAD/data/v3.2.2_test.csv')
    test = test.drop(columns=['Unnamed: 0'])
    return train, test
    

def generate_adversarial_examples(model, data, data_labels, non_encoded_data_labels, eps_range):
    losses = []
    accs = []
    epsilons = []

    for epsilon in eps_range:
        epsilons.append(epsilon)
        adv_samples = fast_gradient_method(model_fn=model, x= data, eps=epsilon, norm=np.inf)
        loss, acc = model.evaluate(verbose=1, x = adv_samples, y = data_labels, batch_size=10)
        losses.append(loss)
        accs.append(acc)

    return epsilons, losses, accs


def plot_epsilons(epsilons, losses, accs, flag):
  plt.style.use('ggplot')
  plt.clf()
  plt.figure(figsize=(9,6))
  plt.xlabel("Epsilon:")
  plt.title("Epsilon range {} to {}".format(eps_range[0],eps_range[-1]))
  
  if flag == 'loss':
    plt.plot(eps_range, losses, color = 'r', marker='.', label='loss')
  if flag == 'acc':
    plt.plot(eps_range, accs, color='g', marker='.', label='accuracy')
  if flag == 'all':
    plt.plot(eps_range, losses, color = 'r', marker='.', label='loss')
    plt.plot(eps_range, accs, color='g', marker='.', label='accuracy')
    
  plt.legend(loc='middle left')
  plt.show()
  plt.savefig('epsilon_loss_accuracy.png')



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

    #Epsilon range
    eps_range = np.concatenate([np.arange(0.0001, 0.001, 0.0001),
                                np.arange(0.001, 0.01, 0.001),
                                np.arange(0.01, 0.1, 0.01)])
    plt.plot(eps_range) 
    
    
    #Generate adversarial examples over a range of epsilons
    epsilons, losses, accs = generate_adversarial_examples(new_model, test_data, test_labels, non_encoded_test_labels, eps_range)
    
    #Visualize the losses and accuracy of adversarial examples on the clean model
    plot_epsilons(epsilons, losses, accs, flag='loss')
    plot_epsilons(epsilons, losses, accs, flag='acc')
    plot_epsilons(epsilons, losses, accs, flag='all')
    

    
    
    

    







    
