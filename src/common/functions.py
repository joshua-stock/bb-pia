import operator

import numpy as np
import pandas as pd
import os
from keras.layers import RandomFlip, Conv2D, GroupNormalization, MaxPooling2D, Dense, Flatten
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam, SGD
from keras import Sequential, Input
from keras.src.saving.saving_api import load_model, save_model
from keras.utils import set_random_seed
from numpy import random
import keras
import tensorflow as tf


class DatasetWithForcedDistribution:
    def __init__(self, sensitive_attribute_name, distribution, X_train, X_test, y_train, y_test, sensitive,
                 sensitive_t):
        """
        Represents a dataset with a forced distribution of a sensitive attribute.

        Args:
            sensitive_attribute_name (str): The name of the sensitive attribute.
            distribution (float): The target distribution of the sensitive attribute.
            X_train (pandas.DataFrame): The training features.
            X_test (pandas.DataFrame): The testing features.
            y_train (pandas.Series): The training labels.
            y_test (pandas.Series): The testing labels.
            sensitive (numpy.ndarray): The values of the sensitive attribute in the training set.
            sensitive_t (numpy.ndarray): The values of the sensitive attribute in the testing set.
        """
        self.sensitive_attribute_name = sensitive_attribute_name
        self.distribution = distribution
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sensitive = sensitive
        self.sensitive_t = sensitive_t


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def drop_index(df, idx):
    """
    Drop rows from a DataFrame based on their index.

    Args:
        df (pandas.DataFrame): The DataFrame to drop rows from.
        idx (int or list): The index or indices of the rows to drop.

    Returns:
        pandas.DataFrame: The DataFrame with the specified rows dropped.
    """
    return df.reset_index().drop(index=idx).drop(columns=["index"])


def find_indices_to_drop(sensitive, target_distribution):
    length = len(sensitive)
    indices_to_drop = []

    def current_dist(sensitive_value=1):
        if sensitive_value == 1:
            return (pd.Series(sensitive).value_counts()[1] - len(indices_to_drop)) / (length - len(indices_to_drop))
        else:
            return (pd.Series(sensitive).value_counts()[1]) / (length - len(indices_to_drop))

    if current_dist() > target_distribution:
        comp = operator.gt
        sensitive_value_to_delete = 1
    else:
        comp = operator.lt
        sensitive_value_to_delete = 0

    i = 0
    while comp(current_dist(sensitive_value_to_delete), target_distribution):
        if i >= length:  # If i reaches the length of the sensitive array, raise an exception
            raise ValueError("Unable to reach target distribution. Not enough entries with the sensitive value to delete.")
        if sensitive[i] == sensitive_value_to_delete:
            # Calculate the difference between the current and target distributions
            diff = abs(current_dist(sensitive_value_to_delete) - target_distribution)
            # If the difference is large, drop multiple entries at once
            if diff > 0.1:
                indices_to_drop.extend([i + j for j in range(10) if i + j < length])
                i += 10
            else:
                indices_to_drop.append(i)
                i += 1
        else:
            i += 1

    return indices_to_drop


def get_lucasnet_sequence(num_classes, input_shape):
    groups = 32
    return [
        Input(shape=input_shape),
        RandomFlip(
            "horizontal",
            seed=42,
        ),
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
        ),
        GroupNormalization(groups=groups),
        MaxPooling2D(2, 2),
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
        ),
        GroupNormalization(groups=groups),
        MaxPooling2D(2, 2),
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
        ),
        GroupNormalization(groups=groups),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation="relu"),
        GroupNormalization(groups=groups),
        Dense(num_classes, activation="softmax"),
    ]

def get_lucasnet_model(num_classes, input_shape):
    return Sequential(get_lucasnet_sequence(num_classes, input_shape))



def compile_lucasnet(model):
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def fit_lucasnet(X_train, y_train, X_test, y_test, batch_size=32, epochs=5, verbose=0, input_shape=(64, 64, 3), num_classes=2):
    model = get_lucasnet_model(num_classes, input_shape)
    model = compile_lucasnet(model)
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=None if X_test is None else (X_test, y_test),
        verbose=verbose
    )
    return model, history


def train_and_generate_output(X_train, y_train, shadow_input, load_model_path, save_model_path, model_no, input_shape, num_classes):
    if os.path.isfile(f"{load_model_path}{model_no}.keras"):
        print(f"Loading model {model_no}")
        shadow_model = load_model(f"{load_model_path}{model_no}.keras")
    else:
        random.seed(model_no)
        set_random_seed(model_no)
        shadow_model, _ = fit_lucasnet(X_train, y_train, X_test=None, y_test=None, input_shape=input_shape, num_classes=num_classes)
        if save_model_path is not None:
            shadow_model.save(f"{save_model_path}{model_no}.keras")
    # To save space, we convert the output to float16
    output = np.array(shadow_model.predict(shadow_input, verbose=0)).astype(np.float16)
    # from an information theoretic perspective, the last column is redundant
    return output[:, 0:output.shape[1]-1]


def generate_shadow_model_outputs(dataset: DatasetWithForcedDistribution, shadow_input, load_model_path, save_model_path, n_shadow_models=100, use_test_data=False, input_shape=(64, 64, 3), num_classes=2):
    if use_test_data:
        X = dataset.X_test
        y = dataset.y_test
    else:
        X = dataset.X_train
        y = dataset.y_train

    #parallel_results_generator = Parallel(n_jobs=20)(
    #    delayed(train_and_generate_output)(X, y, shadow_input, save_model_path, i) for i in range(n_shadow_models))
    #outputs = list(parallel_results_generator)
    outputs = [train_and_generate_output(X, y, shadow_input, load_model_path, save_model_path, i, input_shape, num_classes) for i in range(n_shadow_models)]
    outputs = np.array([o.flatten() for o in outputs])
    return outputs


def train_shadow_models(test_run, n_shadow_models, distributed_datasets, model_input, input_shape, num_classes, base_path, save_models=True):
    for ds in distributed_datasets:
        print(f"now generating {ds.distribution}...")
        load_model_path = f"{base_path}/models/shadow_models/{str(ds.distribution)}/{'test' if test_run else 'train'}/"
        if save_models:
            save_model_path = f"{base_path}/models/shadow_models/{str(ds.distribution)}/{'test' if test_run else 'train'}/"
            ensure_path_exists(save_model_path)
        else:
            save_model_path = None
        outputs = generate_shadow_model_outputs(ds, model_input, load_model_path, save_model_path, n_shadow_models=n_shadow_models, use_test_data=test_run, input_shape=input_shape, num_classes=num_classes)
        adv_df = pd.DataFrame(outputs)
        adv_df["y"] = np.repeat(ds.distribution, n_shadow_models)
        save_data_path = f"{base_path}/data/shadow_model_outputs/{str(ds.distribution)}/"
        ensure_path_exists(save_data_path)
        adv_df.to_csv(f"{save_data_path}{'test' if test_run else 'train'}.csv", index=False)


class DefendingModel(keras.Sequential):

    def __init__(self, adversary, adversary_target, input_for_adversary, training_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adversary = adversary
        self.adversary_target = tf.convert_to_tensor([[adversary_target]])
        self.input_for_adversary = input_for_adversary
        self.training_lambda = training_lambda
        self.adversary_metric = keras.metrics.Mean(name='adversary_prediction')

    @tf.function
    def train_step(self, data):
        x, y = data
        if self.training_lambda > 0:
            with tf.GradientTape() as tape:
                # DEFENSE
                y_pred_for_adv = self(self.input_for_adversary, training=True)
                # last column of prediction is redundant
                num_columns = y_pred_for_adv.shape[1]-1
                y_pred_for_adv = y_pred_for_adv[:, 0:num_columns]
                y_pred_for_adv = Flatten()(y_pred_for_adv)
                # reshape as model input
                my_x = tf.reshape(y_pred_for_adv, (1, y_pred_for_adv.shape[0]*y_pred_for_adv.shape[1]))
                # get adversary prediction
                adv_pred = self.adversary(my_x, training=False)
                # TRAINING
                y_pred = self(x, training=True)

                loss_adv = keras.losses.mean_squared_error(self.adversary_target, adv_pred)
                loss_train = self.compute_loss(x, y, y_pred)
                combined_loss = (1 - self.training_lambda) * loss_train + self.training_lambda * loss_adv

            #self._loss_tracker.update_state(combined_loss)
            #combined_loss = self.optimizer.scale_loss(combined_loss)
            gradients = tape.gradient(combined_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            y_pred_for_adv = self(self.input_for_adversary, training=True)
            my_x = tf.reshape(y_pred_for_adv[:, 0], (1, y_pred_for_adv.shape[0]))
            adv_pred = self.adversary(my_x, training=False)
            with tf.GradientTape() as tape:
                # TRAINING
                y_pred = self(x, training=True)
                combined_loss = self.compute_loss(x, y, y_pred)

            self._loss_tracker.update_state(combined_loss)
            combined_loss = self.optimizer.scale_loss(combined_loss)
            gradients = tape.gradient(combined_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update adversary prediction metric
        self.adversary_metric.update_state(adv_pred)

        # Update model loss
        metrics = self.compute_metrics(x, y, y_pred, sample_weight=None)
        metrics.update({'adversary_prediction': self.adversary_metric.result()})
        return metrics

    def save_inner_model(self, filepath):
        seq = keras.Sequential(self.layers)
        seq = compile_categorical_model(seq)
        save_model(seq, filepath)


def get_defending_lucasnet_model(adversary, adversary_target, input_for_adversary, training_lambda, num_classes, input_shape):
    return DefendingModel(adversary, adversary_target, input_for_adversary, training_lambda, get_lucasnet_sequence(num_classes, input_shape))


def compile_categorical_model(model):
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


class DefendingFairnessModel(keras.Sequential):
    def __init__(self, pia_adversary, adv_input, sensitive, training_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pia_adversary = pia_adversary
        self.adv_input = adv_input
        self.sensitive = sensitive
        self.training_lambda = training_lambda
        self.adversary_predictions = []
        self.p_rule_values = []

        # Initialize TensorFlow variables for the fairness adversary
        self.init_fairness_adversary()

    @staticmethod
    def p_rule(y_pred, z_values):
        threshold = 0.5
        y_z_1 = tf.cast(y_pred[z_values == 1] > threshold, dtype=tf.float16)
        y_z_0 = tf.cast(y_pred[z_values == 0] > threshold, dtype=tf.float16)
        odds = tf.reduce_mean(y_z_1) / tf.reduce_mean(y_z_0)
        return tf.minimum(odds, 1/odds)

    def init_fairness_adversary(self):
        adv = keras.Sequential([
            # num classes
            Input(shape=(2,)),
            Dense(5, activation='relu'), # 2-relu-SGD mit LR 0.05 funktioniert
            Dense(2, activation='softmax')
        ])
        adv.compile(optimizer=Adam(), loss=CategoricalCrossentropy, metrics=['accuracy'])
        self.init_adv_weights = adv.get_weights()
        self.adversary = adv

    def train_fairness_adversary(self, y_pred_for_adv, i):
        with tf.GradientTape() as tape:
            # Forward pass
            adv_pred = self.adversary(y_pred_for_adv)
            # Compute the loss value
            adv_loss = tf.keras.losses.CategoricalCrossentropy()(self.sensitive, adv_pred)

        # print every 10th iteration
        if i % 50 == 0:
            tf.print("adv. loss: ", adv_loss)
        # Compute gradients
        trainable_vars = self.adversary.trainable_variables
        gradients = tape.gradient(adv_loss, trainable_vars)

        # Update weights
        self.adversary.optimizer.apply_gradients(zip(gradients, trainable_vars))

    #def train_pia_adversary(

    @tf.function
    def train_step(self, data):
        x, y = data

        y_pred_for_adv = self(x, training=False)

        # reset adversary weights
        self.adversary.set_weights(self.init_adv_weights)
        # Train the fairness adversary
        for i in range(151):
            self.train_fairness_adversary(y_pred_for_adv, i)

        with tf.GradientTape() as tape:
            y_pred_for_adv = self(x, training=False)
            #y_pred_for_adv = tf.expand_dims(tf.cast(y_pred_for_adv, dtype=tf.float32), axis=1)

            adv_pred = self.adversary(y_pred_for_adv, training=False)
            adv_loss = tf.keras.losses.CategoricalCrossentropy()(self.sensitive, adv_pred)

            y_pred = self(x, training=True)
            loss_train = self.compute_loss(x, y, y_pred)
            combined_loss = loss_train * (1-self.training_lambda) - self.training_lambda * adv_loss

        gradients = tape.gradient(combined_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = self.compute_metrics(x, y, y_pred, sample_weight=None)

        return metrics

    def save_inner_model(self, filepath):
        seq = keras.Sequential(self.layers)
        seq = compile_categorical_model(seq)
        save_model(seq, filepath)



def get_fairness_lucasnet_model(pia_adversary, adversary_input, sensitive, training_lambda, num_classes, input_shape):
    return DefendingFairnessModel(pia_adversary, adversary_input, sensitive, training_lambda, get_lucasnet_sequence(num_classes, input_shape))


def set_seeds(training_lambda, run, distribution):
    seed = int(training_lambda * 1000 + distribution * 100 + run)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

