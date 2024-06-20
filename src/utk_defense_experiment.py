import os
import random
import numpy as np
import tensorflow as tf
import csv
import keras
from utk_functions import get_lbfw_dataset, get_distributed_utk_sets
from datetime import datetime
from common.functions import get_defending_lucasnet_model, compile_categorical_model, ensure_path_exists

affinity_mask = set()
for i in range(7):
    affinity_mask.add(7+i)
os.sched_setaffinity(0, affinity_mask)

resultspath = "utkface/results/"
ensure_path_exists(resultspath)
adversary = keras.models.load_model('utkface/models/adv_v2_0.63_r2.keras')
model_input = get_lbfw_dataset()
#distributed_datasets = get_distributed_utk_sets()
lambdas = [0.01, 0.2, 0.15]#, 0.15, 0.1, 0.15, 0.2]
runs = 10
time = datetime.now().strftime('%Y%m%d-%H%M%S')
resultsfile = f"{resultspath}utk_defense_v2_results-{time}.csv"


def set_seeds(training_lambda, run, distribution):
    seed = int(training_lambda * 1000 + distribution * 100 + run)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

for run in range(runs):
    for training_lambda in lambdas:
        for dis in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            save_path = f"utkface/models/defense/"
            ensure_path_exists(save_path)
            complete_save_path = f"{save_path}utkdef-v2-ds{dis}-l{training_lambda}-run{run}.keras"
            if os.path.isfile(complete_save_path):
                print(f"skipping {complete_save_path}")
                continue
                      
            ds = get_distributed_utk_sets([dis])[0]
                      
            logs = f"logs/utkdef-ds{ds.distribution}-l{training_lambda}-run{run}-{time}"
            print(f"Now training ds{ds.distribution}-l{training_lambda}-run{run}-{time}")
            #tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
            #                                                 update_freq=20)

            set_seeds(training_lambda, run, ds.distribution)

            model = get_defending_lucasnet_model(
                adversary=adversary,
                adversary_target=0.5,
                input_for_adversary=model_input,
                training_lambda=training_lambda,
                num_classes=2,
                input_shape=(64, 64, 3))

            model = compile_categorical_model(model)
            model_history = model.fit(
                ds.X_test,
                ds.y_test,
                epochs=4,
                validation_data=(ds.X_train, ds.y_train),
                batch_size=32,
                #callbacks=[tboard_callback],
                verbose=0)
            model.save_inner_model(complete_save_path)

            output = model.predict(model_input)
            formatted_input = output[:, 0].reshape(1, output.shape[0])
            adv_out = adversary(formatted_input).numpy().flatten()[0]
            test_acc = model.evaluate(ds.X_train, ds.y_train)

            with open(resultsfile, 'a', newline='') as csvfile:
                resultwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                resultwriter.writerow([training_lambda, ds.distribution, run, test_acc, adv_out])
