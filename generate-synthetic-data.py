from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from pia_functions import data_train_test
import pandas as pd
import numpy as np


def generate_synthetic_data(save_path, use_existing_model: bool = True, output_size: int = 10000):
    """
    Generate synthetic data using CTGANSynthesizer.

    Args:
        save_path (str): The path to save the generated synthetic data.
        use_existing_model (bool): Flag indicating whether to use an existing model or train a new one.
        output_size (int): Number of rows to generate in the synthetic dataset.

    Returns:
        pd.DataFrame: The generated synthetic dataset.
    """
    np.random.seed(1)

    X_train, X_test, _, _, _, _ = data_train_test()
    if not use_existing_model:
        metadata = SingleTableMetadata()
        to_fit = pd.DataFrame(np.concatenate((X_train, X_test)), columns=[str(i) for i in range(79)])
        metadata.detect_from_dataframe(data=to_fit)

        syn_model = CTGANSynthesizer(metadata)
        syn_model.fit(to_fit)
        syn_model.save('syn_model')
    else:
        # in case there are problems with loading the model, try downgrading via pip install sdv==1.5
        syn_model = CTGANSynthesizer.load('syn_model')

    sampled = syn_model.sample(num_rows=output_size)

    # save data
    sampled.to_csv(save_path, index=False, header=X_train.columns)

    return sampled


if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(save_path = "data/syn_data-new.csv", use_existing_model=True, output_size=10000)
    print("generating synthetic data done.")
