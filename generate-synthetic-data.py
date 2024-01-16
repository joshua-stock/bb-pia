from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from pia_functions import data_train_test
import pandas as pd
import numpy as np

use_existing_model = True
output_size = 10000
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
sampled.to_csv("data/syn_data-new.csv", index=False, header=X_train.columns)
