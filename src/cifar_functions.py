import keras
import tensorflow as tf
import numpy as np
from common.functions import find_indices_to_drop, DatasetWithForcedDistribution
from sklearn.model_selection import train_test_split


def get_animal_classes(y):
    animal_classes = [2, 3, 4, 5, 6, 7]  # 'bird', 'cat', 'deer', 'dog', 'frog', 'horse'
    return np.isin(y, animal_classes).flatten()


def format_y(y):
    return tf.one_hot(tf.convert_to_tensor(y.flatten(), dtype=tf.dtypes.int32), depth=10)


def data_train_test_cifar(train_size=0.5):
    cifar10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Concatenate x_train with x_test and y_train with y_test
    x, y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    sensitive = get_animal_classes(y)

    # normalization
    x = x / 255.0

    # Split the concatenated data into training and testing sets
    x_train, x_test, y_train, y_test, sens_train, sens_test = train_test_split(x,
                                                                               y,
                                                                               sensitive,
                                                                               train_size=train_size,
                                                                               shuffle=True,
                                                                               stratify=y,
                                                                               random_state=0)

    y_train, y_test = format_y(y_train), format_y(y_test)

    return x_train, x_test, y_train, y_test, sens_train, sens_test


def drop_index_cifar(ar, idx):
    return np.delete(ar, idx, axis=0)


def get_distributed_cifar_sets(distributions=None):
    if distributions is None:
        distributions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_datasets = []
    for dist in distributions:
        print(f'now computing dist {dist}')
        np.random.seed(1)
        X_train, X_test, y_train, y_test, sensitive, sensitive_t = data_train_test_cifar()

        indices_to_drop = find_indices_to_drop(sensitive, dist)
        indices_to_drop_t = find_indices_to_drop(sensitive_t, dist)

        df = DatasetWithForcedDistribution(
            sensitive_attribute_name='animals',
            distribution=dist,
            X_train=drop_index_cifar(X_train, idx=indices_to_drop),
            X_test=drop_index_cifar(X_test, idx=indices_to_drop_t),
            y_train=drop_index_cifar(y_train, idx=indices_to_drop),
            y_test=drop_index_cifar(y_test, idx=indices_to_drop_t),
            sensitive=drop_index_cifar(sensitive, idx=indices_to_drop),
            sensitive_t=drop_index_cifar(sensitive_t, idx=indices_to_drop_t)
        )
        all_datasets.append(df)
    return all_datasets


def get_cifar_input_set():
    cifar100 = keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x, y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)

    # The following indices that have the least similarity to the CIFAR-10 dataset
    class_indices_to_include = [20, 23, 26, 32, 34, 35, 41, 42, 43, 46, 47, 52, 53, 55, 61, 71, 77, 82, 83, 84, 87, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    mask = np.isin(y, class_indices_to_include).flatten()
    x, y = x[mask], y[mask]

    # sample to shrink the dataset
    x_shrunk, _, _, _ = train_test_split(x, y, train_size=0.1, random_state=0, stratify=y)

    return x_shrunk


