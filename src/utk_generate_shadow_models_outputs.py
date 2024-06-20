from utk_functions import get_lbfw_dataset, get_distributed_utk_sets
from common.functions import train_shadow_models
import os

affinity_mask = set()
for i in range(5):
    affinity_mask.add(15+i)
os.sched_setaffinity(0, affinity_mask)

if __name__ == "__main__":
    base_path = "utkface"  # for saving models and output data
    test_run = True
    n_shadow_models = 50
    distributions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    save_models = True
    input_shape = (64, 64, 3)
    num_classes = 2
    model_input = get_lbfw_dataset()
    distributed_datasets = get_distributed_utk_sets(distributions=distributions)
    print(f"Generating {n_shadow_models} {'test' if test_run else 'train'} shadow models...")
    train_shadow_models(test_run, n_shadow_models, distributed_datasets, model_input, input_shape, num_classes, base_path, save_models)
    print(f"Generated {n_shadow_models} {'test' if test_run else 'train'} shadow models. DONE")
