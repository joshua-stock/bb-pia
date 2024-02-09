import numpy as np
import pandas as pd

from utk_functions import get_lbfw_dataset, get_distributed_utk_sets, generate_shadow_model_outputs
from common.functions import ensure_path_exists


def train_shadow_models(test_run, n_shadow_models, distributions, model_input, save_models=True):
    distributed_datasets = get_distributed_utk_sets(distributions=distributions)
    #all_shadow_outputs = []

    for ds in distributed_datasets:
        print(f"now generating {ds.distribution}...")
        load_model_path = f"/dev/shm/utkface/models/shadow_models/{str(ds.distribution)}/{'test' if test_run else 'train'}/"
            #f"utkface/models/shadow_models/{str(ds.distribution)}/{'test' if test_run else 'train'}/"
        if save_models:
            save_model_path = f"/dev/shm/utkface/models/shadow_models/{str(ds.distribution)}/{'test' if test_run else 'train'}/"
            ensure_path_exists(save_model_path)
        else:
            save_model_path = None
        outputs = generate_shadow_model_outputs(ds, model_input, load_model_path, save_model_path, n_shadow_models=n_shadow_models, use_test_data=test_run)
        adv_df = pd.DataFrame(outputs)
        adv_df["y"] = np.repeat(ds.distribution, n_shadow_models)
        save_data_path = f"utkface/data/shadow_model_outputs/{str(ds.distribution)}/"
        ensure_path_exists(save_data_path)
        adv_df.to_csv(f"{save_data_path}{'test' if test_run else 'train'}.csv", index=False)
        #all_shadow_outputs.append(outputs)

    #adv_df = pd.DataFrame(np.array(np.concatenate(all_shadow_outputs)))
    #adv_df["y"] = np.concatenate(([np.repeat(d, n_shadow_models) for d in distributions]))
    #adv_df.to_csv(f"utkface/data/shadow_model_outputs_{'test' if test_run else 'train'}.csv", index=False)


if __name__ == "__main__":
    test_run = False
    n_shadow_models = 200
    distributions = [0.7, 0.8, 0.9]
    save_models = False
    model_input = get_lbfw_dataset()
    print(f"Generating {n_shadow_models} {'test' if test_run else 'train'} shadow models...")
    train_shadow_models(test_run, n_shadow_models, distributions, model_input, save_models)
    print(f"Generated {n_shadow_models} {'test' if test_run else 'train'} shadow models. DONE")
