import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from common.functions import find_indices_to_drop, drop_index, DatasetWithForcedDistribution


def data_train_test(train_size=0.5):
    """
    Split the adult dataset into training and testing sets.

    Args:
        train_size (float): The proportion of the dataset to include in the training set.

    Returns:
        tuple: A tuple containing the training and testing features and labels, as well as the values of the sensitive attribute.
    """
    adult_df = pd.read_csv("adult/data/census_data_oh.csv").drop(columns=["sex_Female", "income_<=50K"])
    y = adult_df["income_>50K"]
    X_train, X_test, y_train, y_test = train_test_split(adult_df.drop(columns=["income_>50K"]), y, train_size=train_size, random_state=0)
    sensitive = X_train["sex_Male"].reset_index(drop=True)
    sensitive_t = X_test["sex_Male"].reset_index(drop=True)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True), sensitive, sensitive_t


def get_distributed_adult_sets(distributions=None):
    """
    Generate datasets with forced distributions of the sensitive attribute.

    Args:
        distributions (list): A list of target distributions. If None, default distributions will be used.

    Returns:
        list: A list of DatasetWithForcedDistribution objects representing the generated datasets.
    """
    if distributions is None:
        distributions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_datasets = []
    for dist in distributions:
        print(f'now computing dist {dist}')
        np.random.seed(1)
        X_train, X_test, y_train, y_test, sensitive, sensitive_t = data_train_test()

        indices_to_drop = find_indices_to_drop(sensitive, dist)
        indices_to_drop_t = find_indices_to_drop(sensitive_t, dist)

        df = DatasetWithForcedDistribution(
            sensitive_attribute_name='sex',
            distribution=dist,
            X_train=drop_index(X_train, idx=indices_to_drop),
            X_test=drop_index(X_test, idx=indices_to_drop_t),
            y_train=drop_index(y_train, idx=indices_to_drop).values.flatten(),
            y_test=drop_index(y_test, idx=indices_to_drop_t).values.flatten(),
            sensitive=np.squeeze(drop_index(sensitive, idx=indices_to_drop)),
            sensitive_t=np.squeeze(drop_index(sensitive_t, idx=indices_to_drop_t))
        )
        all_datasets.append(df)
    return all_datasets


def train_gradient_boosting_shadow_model(X_train, y_train, random_state):
    """
    Train a gradient boosting shadow model.

    Args:
        X_train (pandas.DataFrame): The training features.
        y_train (pandas.Series): The training labels.
        random_state (int): The random state for reproducibility.

    Returns:
        GradientBoostingClassifier: The trained gradient boosting model.
    """
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, max_features=90, random_state=random_state, min_impurity_decrease=0.0,
                                    min_samples_leaf=2, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0)
    return gb.fit(X_train, y_train)


def train_and_generate_output(X_train, y_train, shadow_input, output_probability, random_state=0):
    """
    Train a shadow model and generate its outputs.

    Args:
        X_train (pandas.DataFrame): The training features.
        y_train (pandas.Series): The training labels.
        shadow_input (pandas.DataFrame): The input to the shadow model.
        output_probability (bool): Whether to output probabilities or predictions.
        random_state (int): The random state for reproducibility.

    Returns:
        numpy.ndarray: The outputs of the shadow model.
    """
    shadow_model = train_gradient_boosting_shadow_model(X_train, y_train, random_state=random_state)
    if output_probability:
        output = shadow_model.predict_proba(shadow_input)
    else:
        output = shadow_model.predict(shadow_input)
    return output[:, 0]


def generate_shadow_model_outputs(dataset: DatasetWithForcedDistribution, shadow_input, n_shadow_models=100, use_test_data=False, output_probability=True):
    """
    Generate outputs of shadow models for a given dataset.

    Args:
        dataset (DatasetWithForcedDistribution): The dataset with forced distribution of the sensitive attribute.
        shadow_input (pandas.DataFrame): The input to the shadow models.
        n_shadow_models (int): The number of shadow models to generate outputs from.
        use_test_data (bool): Whether to use the testing data or training data.
        output_probability (bool): Whether to output probabilities or predictions.

    Returns:
        list: A list of outputs from the shadow models.
    """
    if use_test_data:
        X = dataset.X_test
        y = dataset.y_test
    else:
        X = dataset.X_train
        y = dataset.y_train

    parallel_results_generator = Parallel(n_jobs=20)(
        delayed(train_and_generate_output)(X, y, shadow_input, output_probability, i) for i in range(n_shadow_models))
    outputs = list(parallel_results_generator)
    return outputs
