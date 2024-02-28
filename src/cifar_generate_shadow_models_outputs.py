from cifar_functions import get_distributed_cifar_sets, get_cifar_input_set
from common.functions import train_shadow_models


if __name__ == "__main__":
    base_path = "cifar"  # for saving models and output data
    test_run = False
    n_shadow_models = 200
    distributions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    save_models = False
    input_shape = (32, 32, 3)
    num_classes = 10
    model_input = get_cifar_input_set()
    distributed_datasets = get_distributed_cifar_sets(distributions=distributions)
    print(f"Generating {n_shadow_models} {'test' if test_run else 'train'} shadow models...")
    train_shadow_models(test_run, n_shadow_models, distributed_datasets, model_input, input_shape, num_classes, base_path, save_models)
    print(f"Generated {n_shadow_models} {'test' if test_run else 'train'} shadow models. DONE")
