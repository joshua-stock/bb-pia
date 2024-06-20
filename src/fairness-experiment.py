import json

import keras
from keras.src.layers import Flatten
from keras.src.losses import CategoricalCrossentropy
from keras.src.optimizers import Adam
from utk_functions import get_distributed_utk_sets, get_lbfw_dataset
from common.functions import get_fairness_lucasnet_model, ensure_path_exists, set_seeds
import tensorflow as tf


distributions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
resultspath = 'utkface/results/fairness-3/'
epochs = 18
early_stopping = False

input_set = get_lbfw_dataset()
datasets = get_distributed_utk_sets(distributions)

manual_adversary = keras.Sequential()
manual_adversary.add(keras.Input(shape=(13233,)))
manual_adversary.add(keras.layers.Dense(10, activation='relu'))
manual_adversary.add(keras.layers.Dense(5, activation='relu'))
manual_adversary.add(keras.layers.Dense(1))
manual_adversary.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError(), metrics=[keras.metrics.R2Score()])
manual_adversary.load_weights("utkface/models/manual_tuning_checkpoints-2/keras.weights.h5")

for run in range(10):
    for dataset in datasets:
        for training_lambda in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
            print(f"now running run {run}, dist {dataset.distribution}, training lambda {training_lambda}")
            sensitive_categorical = dataset.sensitive  # tf.keras.utils.to_categorical(dataset.sensitive)
            model = get_fairness_lucasnet_model(
                manual_adversary,
                input_set,
                sensitive=sensitive_categorical,
                training_lambda=training_lambda,
                num_classes=2,
                input_shape=(64, 64, 3))
            model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])#, model.p_rule_metric, model.adversary_metric])

            patience = 5
            decrease = 0
            best_val_acc = 0
            best_model_weights = None
            my_history = {
                'accuracy': [],
                'val_accuracy': [],
                'p_rule': [],
                'adversary_prediction': []
            }

            set_seeds(training_lambda, run, dataset.distribution)

            for i in range(epochs):
                hist = model.fit(
                    dataset.X_train,
                    dataset.y_train,
                    epochs=1,
                    validation_data=(dataset.X_test, dataset.y_test),
                    batch_size=dataset.X_train.shape[0],
                    verbose=0
                    #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)]
                )

                # check pia adversary output
                y_pred_for_adv = model(model.adv_input)
                # last column of prediction is redundant
                num_columns = y_pred_for_adv.shape[1]-1
                y_pred_for_adv = y_pred_for_adv[:, 0:num_columns]
                y_pred_for_adv = Flatten()(y_pred_for_adv)
                # reshape as model input
                my_x = tf.reshape(y_pred_for_adv, (1, y_pred_for_adv.shape[0]*y_pred_for_adv.shape[1]))

                adversary_prediction = model.pia_adversary(my_x).numpy()[0][0]
                p_rule_value = model.p_rule(model(dataset.X_train, training=False), model.sensitive)

                val_acc = hist.history['val_accuracy'][-1]
                print(f"ROUND {i} acc: {hist.history['accuracy'][-1]} val_acc: {val_acc} + p_rule: {p_rule_value} adv_pred: {adversary_prediction}")

                for k, v in zip(my_history.keys(), [hist.history['accuracy'][-1], val_acc, float(p_rule_value), float(adversary_prediction)]):
                    my_history[k].append(v)

                if early_stopping:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        decrease = 0
                        best_model_weights = model.get_weights()
                    else:
                        #model.set_weights(best_model_weights)
                        decrease += 1
                        if decrease >= patience:
                            break
                            
            print("STOPPING")
            ensure_path_exists(resultspath)
            with open(f'{resultspath}result-l{training_lambda}-d{dataset.distribution}-r{run}.json', 'w') as json_file:
                json.dump(my_history, json_file)

print("DONE")
