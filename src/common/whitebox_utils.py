import keras
import numpy as np
import tensorflow as tf
import re
#import multiprocessing as mp
#from joblib import Parallel, delayed

def do_read_single_model_params(m):
    layer_indices_for_adv = [
        1, #conv2d
        4, #conv2d_1
        7, #conv2d_2
        11, #dense
        13 #dense_1
    ]
    return [m.layers[i].weights for i in layer_indices_for_adv]


def transform_input_name(name, index):
    # name cannot contain /
    result = name.replace("/", "-")
    # use standardized index number - add 0 if it does not exist
    result = re.sub("(_([0-9])+)?-", f"_{index}-", result)
    return result


def read_single_model_params(path):
    model = keras.models.load_model(path)
    return do_read_single_model_params(model)


def read_mult_model_params_mp(paths):
    # JOBLIB
    #parallel_results_generator = Parallel(n_jobs=5)(
    #    delayed(read_single_model_params)(p) for path in paths)
    #wb = list(parallel_results_generator)
    
    # MP
    #pool = mp.Pool(nr_cpu)
    #w_b = pool.map(read_single_model_params, paths)
    #pool.close()
    
    # SEQUENTIALLY:
    w_b = [read_single_model_params(p) for p in paths]
    
    return w_b


def load_model_params(my_models_per_y, ys, base_path, train_or_test):
    paths = [f"{base_path}/{j}/{train_or_test}/{i}.keras" for j in ys for i in range(my_models_per_y)]
    return read_mult_model_params_mp(paths)


def construct_dataset(wb, y):
    def my_generator():
        for w, my_y in zip(wb, y):
            yield {transform_input_name(single_weight.path, i): single_weight for row, i in zip(w, range(len(w))) for single_weight in row}, [my_y]
    
    out_types = {transform_input_name(single_weight.path, i): tf.float32 for w, i in zip(wb[1], range(len(wb[1]))) for single_weight in w}
    out_shapes = {transform_input_name(single_weight.path, i): single_weight.shape for w, i in zip(wb[1], range(len(wb[1]))) for single_weight in w}
    return tf.data.Dataset.from_generator(my_generator, output_types=(out_types, tf.float32), output_shapes=(out_shapes, (1)))


def get_dataset(models_per_y, base_path, train_or_test='train', all_y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    wb = load_model_params(models_per_y, all_y, base_path, train_or_test)
    y = np.repeat(all_y, models_per_y)
    return construct_dataset(wb, y)


@keras.saving.register_keras_serializable()
class ConvSplitter(keras.layers.Layer):
    def __init__(self, neurons, **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
    
    #overwrite get_config for being able to save and load model
    def get_config(self):
        config = {
            "neurons" : self.neurons
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        return [inputs[:, :, :, :, i] for i in range(self.neurons)]

    
@keras.saving.register_keras_serializable()
class WeightsSplitter(keras.layers.Layer):
    def __init__(self, neurons, **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
    
    #overwrite get_config for being able to save and load model
    def get_config(self):
        config = {
            "neurons" : self.neurons
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        return [inputs[:, :, i] for i in range(self.neurons)]

    
@keras.saving.register_keras_serializable()   
class BiasSplitter(keras.layers.Layer):
    def __init__(self, neurons, **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
        
    #overwrite get_config for being able to save and load model
    def get_config(self):
        config = {
            "neurons" : self.neurons
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        return [inputs[:, i] for i in range(self.neurons)]
