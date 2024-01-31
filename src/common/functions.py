import operator
import pandas as pd


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
    """
    Find the indices of rows to drop in order to achieve a target distribution of a sensitive attribute.

    Args:
        sensitive (numpy.ndarray): The values of the sensitive attribute.
        target_distribution (float): The target distribution of the sensitive attribute.

    Returns:
        list: The indices of rows to drop.
    """
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
        if sensitive[i] == sensitive_value_to_delete:
            indices_to_drop.append(i)
        i = i + 1

    return indices_to_drop


