import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def data_train_test_cifar(train_size=0.5):
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Concatenate x_train with x_test and y_train with y_test
    x, y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)

    animal_classes = [2, 3, 4, 5, 6, 7]  # 'bird', 'cat', 'deer', 'dog', 'frog', 'horse'
    sensitive = np.isin(y, animal_classes)

    x = x / 255.0

    # Split the concatenated data into training and testing sets
    x_train, x_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(x, y, sensitive, train_size=train_size, shuffle=True, stratify=y, random_state=0)
        
    y_train, y_test = tf.one_hot(tf.convert_to_tensor(y_train.flatten(), dtype=tf.dtypes.int32), depth=10), tf.one_hot(tf.convert_to_tensor(y_test.flatten(), dtype=tf.dtypes.int32), depth=10)

    return x_train, x_test, y_train, y_test, sensitive_train, sensitive_test
