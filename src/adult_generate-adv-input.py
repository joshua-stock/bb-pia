import pandas as pd
import numpy as np
from adult_functions import get_distributed_adult_sets, generate_shadow_model_outputs


def generate_adv_input(test_run, n_shadow_models, distributions, model_input):
    distributed_datasets = get_distributed_adult_sets(distributions=distributions)
    all_shadow_outputs = []
    for ds in distributed_datasets:
        print(f"now generating {ds.distribution}...")
        outputs = generate_shadow_model_outputs(ds, model_input, n_shadow_models=n_shadow_models, use_test_data=test_run, output_probability=True)
        all_shadow_outputs.append(outputs)

    adv_df = pd.DataFrame(np.array(np.concatenate(all_shadow_outputs)))
    adv_df["y"] = np.concatenate(([np.repeat(d, n_shadow_models) for d in distributions]))
    adv_df.to_csv(f"adult/data/shadow_model_outputs_proba-new{'_test_set' if test_run else ''}.csv", index=False)


if __name__ == "__main__":
    test_run = False
    n_shadow_models = 200
    distributions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_input = pd.read_csv("adult/data/syn_data-new.csv")
    generate_adv_input(test_run, n_shadow_models, distributions, model_input)
