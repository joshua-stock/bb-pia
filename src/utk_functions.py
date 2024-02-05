import os

import numpy as np
from common.functions import find_indices_to_drop, DatasetWithForcedDistribution
from keras.applications import MobileNetV3Small
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers import AdamW
from keras.src.applications.efficientnet import EfficientNetB0
from keras.utils import load_img, img_to_array, to_categorical
from sklearn.model_selection import train_test_split


def get_old_indices(filelist, threshold=60):
    return [1 if int(i.split('_')[0]) >= threshold else 0 for i in filelist]


def get_gender_classes(filelist):
    gender = [i.split('_')[1] for i in filelist]

    result = []
    for i in gender:
        i = int(i)
        if i == 0:
            result.append(0)
        elif i == 1:
            result.append(1)
    return result


def get_utkface_gender_prediction_dataset(files_dir):
    onlyfiles = os.listdir(files_dir)

    gender_classes = get_gender_classes(onlyfiles)

    # convert images to vectors
    X = []
    for file in onlyfiles:
        image_path = os.path.join(files_dir, file)
        rgb_image = load_img(image_path, target_size=(64, 64))
        x = img_to_array(rgb_image)
        #x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X.append(x)


    # normalize data
    #X = X.astype('float32')
    #X /= 255

    Y = to_categorical(gender_classes, num_classes=2)
    X = np.asarray(X)

    return X, Y, onlyfiles


def head_model(base_model, n_classes):
    top_model = base_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(n_classes, activation="softmax")(top_model)
    return top_model


def get_mobilenet_model(num_classes=2):
    mobilenet_mdl = MobileNetV3Small(
        include_top=False,
        input_shape=(64, 64, 3),
        weights="imagenet",
        alpha=0.75
    )
    model_mobilenet_head = head_model(mobilenet_mdl, num_classes)
    model_mobilenet = Model(inputs = mobilenet_mdl.input, outputs = model_mobilenet_head)
    model_mobilenet.compile(
        optimizer=AdamW(learning_rate=0.0001, weight_decay=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model_mobilenet


def get_efficientnet_model(num_classes=2):
    mobilenet_mdl = EfficientNetB0(
        include_top=False,
        input_shape=(64, 64, 3),
        weights="imagenet"
    )
    model_mobilenet_head = head_model(mobilenet_mdl, num_classes)
    model_mobilenet = Model(inputs = mobilenet_mdl.input, outputs = model_mobilenet_head)
    model_mobilenet.compile(
        optimizer=AdamW(learning_rate=0.0001, weight_decay=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model_mobilenet


def fit_mobilenet(X_train, y_train, X_test, y_test, batch_size=32, epochs=12):
    model = get_mobilenet_model()

    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test)
    )
    return model, history


def drop_index_utk(ar, idx):
    return np.delete(ar, idx, axis=0)


def data_train_test_utk(train_size=0.5, utk_root='utkface/data/utkface'):
    X, Y, onlyfiles = get_utkface_gender_prediction_dataset(utk_root)

    old_idx = np.asarray(get_old_indices(onlyfiles, threshold=60))

    # Generate a permutation of indices
    rand_idx = np.random.permutation(len(X))
    shuffled_X = X[rand_idx]
    shuffled_Y = Y[rand_idx]
    shuffled_old_idx = old_idx[rand_idx]


    X_train, X_test, y_train, y_test = train_test_split(
        shuffled_X,
        shuffled_Y,
        train_size=train_size,
        shuffle=False
    )

    sensitive = shuffled_old_idx[0: len(X_train)]
    sensitive_t = shuffled_old_idx[len(X_train):]

    return X_train, X_test, y_train, y_test, sensitive, sensitive_t


def get_distributed_utk_sets(distributions=None):
    """
    Generate datasets with forced distributions of the sensitive attribute.

    Args:
        distributions (list): A list of target distributions. If None, default distributions will be used.

    Returns:
        list: A list of DatasetWithForcedDistribution objects representing the generated datasets.
    """
    if distributions is None:
        distributions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_datasets = []
    for dist in distributions:
        print(f'now computing dist {dist}')
        np.random.seed(1)
        X_train, X_test, y_train, y_test, sensitive, sensitive_t = data_train_test_utk()

        indices_to_drop = find_indices_to_drop(sensitive, dist)
        indices_to_drop_t = find_indices_to_drop(sensitive_t, dist)

        df = DatasetWithForcedDistribution(
            sensitive_attribute_name='age',
            distribution=dist,
            X_train=drop_index_utk(X_train, idx=indices_to_drop),
            X_test=drop_index_utk(X_test, idx=indices_to_drop_t),
            y_train=drop_index_utk(y_train, idx=indices_to_drop),
            y_test=drop_index_utk(y_test, idx=indices_to_drop_t),
            sensitive=drop_index_utk(sensitive, idx=indices_to_drop),
            sensitive_t=drop_index_utk(sensitive_t, idx=indices_to_drop_t)
        )
        all_datasets.append(df)
    return all_datasets


def get_lbfw_dataset(lfw_root='utkface/data/lfw-deepfunneled'):
    X = []
    all_lfw_names = os.listdir(lfw_root)

    for name in all_lfw_names:
        dir_path = os.path.join(lfw_root, name)
        list_images_name = os.listdir(dir_path)

        for image_name in list_images_name:
            image_path = os.path.join(dir_path, image_name)
            rgb_image = load_img(image_path, target_size=(64, 64))
            x = img_to_array(rgb_image)
            x = preprocess_input(x)
            X.append(x)

    return np.asarray(X)