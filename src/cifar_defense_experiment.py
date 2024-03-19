import os
import random
import numpy as np
import tensorflow as tf
import csv
import keras
from cifar_functions import get_cifar_input_set, get_distributed_cifar_sets
from datetime import datetime
from keras.src.layers import Flatten
from common.functions import get_defending_lucasnet_model, compile_categorical_model, ensure_path_exists

affinity_mask = set()
for i in range(10):
    affinity_mask.add(i + 10)
os.sched_setaffinity(0, affinity_mask)

resultspath = "cifar/results/"
ensure_path_exists(resultspath)
adversary = keras.models.load_model('cifar/models/cifar-adv_0.64_test_r2.keras')
model_input = get_cifar_input_set()
distributed_datasets = get_distributed_cifar_sets()
lambdas = [0.2, 0.25, 0.0] #0.05, 0.1, 0.15, 
runs = 10
time = datetime.now().strftime('%Y%m%d-%H%M%S')
resultsfile = f"{resultspath}cifar_defense_results-{time}.csv"


def set_seeds(training_lambda, run, distribution):
    seed = int(training_lambda * 1000 + distribution * 100 + run)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    
for ds in distributed_datasets:
    for training_lambda in lambdas:
        for run in range(runs):
            save_path = f"cifar/models/defense/"
            ensure_path_exists(save_path)
            logs = f"logs/cifardef-ds{ds.distribution}-l{training_lambda}-run{run}-{time}"
            print(f"Now training ds{ds.distribution}-l{training_lambda}-run{run}-{time}")
            tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                             update_freq=5)

            set_seeds(training_lambda, run, ds.distribution)

            model = get_defending_lucasnet_model(
                adversary=adversary,
                adversary_target=0.5,
                input_for_adversary=model_input,
                training_lambda=training_lambda,
                num_classes=10,
                input_shape=(32, 32, 3))

            model = compile_categorical_model(model)
            model_history = model.fit(
                ds.X_test,
                ds.y_test,
                epochs=4,
                validation_data=(ds.X_train, ds.y_train),
                batch_size=32,
                callbacks=[tboard_callback],
                verbose=0)
            model.save_inner_model(f"{save_path}cifardef-ds{ds.distribution}-l{training_lambda}-run{run}.keras")

            output = model.predict(model_input)
            num_columns = output.shape[1]-1
            output = output[:, 0:num_columns]
            output = Flatten()(output)
            # reshape as model input
            my_x = tf.reshape(output, (1, output.shape[0]*output.shape[1]))
            # get adversary prediction
            adv_out = adversary(my_x).numpy().flatten()[0]
            test_acc = model.evaluate(ds.X_train, ds.y_train)

            with open(resultsfile, 'a', newline='') as csvfile:
                resultwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                resultwriter.writerow([training_lambda, ds.distribution, run, test_acc, adv_out])
