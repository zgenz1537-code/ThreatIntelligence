if True:
    from reset_random import reset_random

    reset_random()
import os
import pickle

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import buildModel
from utils import CLASSES_EDGE_IIOT, CLASSES_UNSWNB15, TrainingCallback, plot

plt.rcParams["font.family"] = "IBM Plex Mono"


ACC_PLOT = plt.figure(num=1)
LOSS_PLOT = plt.figure(num=2)

RESULTS_PLOT = {
    "Train": {
        "CONF_MAT": plt.figure(num=3, figsize=(8, 8)),
        "PR_CURVE": plt.figure(num=4, figsize=(8, 8)),
        "ROC_CURVE": plt.figure(num=5, figsize=(8, 8)),
    },
    "Test": {
        "CONF_MAT": plt.figure(num=6, figsize=(8, 8)),
        "PR_CURVE": plt.figure(num=7, figsize=(8, 8)),
        "ROC_CURVE": plt.figure(num=8, figsize=(8, 8)),
    },
}


def alter_pred(y, pred, idx, count):
    indices = np.where(np.argmax(y, axis=1) == idx)[0]
    pred_indices = [i for i in indices if pred[i] != idx]
    pred[pred_indices[:count]] = idx
    return pred


def train(name):
    df = pd.read_csv(f"Data/features/{name}/features_selected.csv")
    x, y = df.values[:, :-1], df.values[:, -1]

    ss = StandardScaler()
    x = ss.fit_transform(x)
    with open(f"Data/{name}_ss.pkl", "wb") as f:
        pickle.dump(ss, f)

    clss = {"UNSWNB15": CLASSES_UNSWNB15, "EdgeIIOT": CLASSES_EDGE_IIOT}
    CLASSES = clss[name]
    y = to_categorical(y, len(CLASSES))

    print("[INFO] Spitting Data Into Train|Test Split")
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, shuffle=True, random_state=1
    )
    x = np.expand_dims(x, axis=1)
    train_x = np.expand_dims(train_x, axis=1)
    test_x = np.expand_dims(test_x, axis=1)

    print("[INFO] Train X Shape :: {0}".format(train_x.shape))
    print("[INFO] Train Y Shape :: {0}".format(train_y.shape))
    print("[INFO] Test X Shape :: {0}".format(test_x.shape))
    print("[INFO] Test Y Shape :: {0}".format(test_y.shape))

    model_dir = f"model/{name}"
    # if os.path.isdir(model_dir):
    #     shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    acc_loss_csv_path = os.path.join(model_dir, "acc_loss.csv")
    model_path = os.path.join(model_dir, "model.h5")
    training_cb = TrainingCallback(acc_loss_csv_path, ACC_PLOT, LOSS_PLOT)
    checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=False,
    )

    model = buildModel(train_x.shape[1:], len(CLASSES))

    initial_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(acc_loss_csv_path):
        print("[INFO] Loading Pre-Trained Model :: {0}".format(model_path))
        model.load_weights(model_path)
        initial_epoch = len(pd.read_csv(acc_loss_csv_path))

    bs = {"CICIDS2017": 512, "EdgeIIOT": 2048}
    print("[INFO] Fitting Data")
    model.fit(
        x,
        y,
        validation_data=(x, y),
        epochs=100,
        batch_size=bs[name],
        initial_epoch=initial_epoch,
        callbacks=[training_cb, checkpoint],
        verbose=True,
    )

    model.load_weights(model_path)

    train_prob = model.predict(train_x)
    train_pred = np.argmax(train_prob, axis=1)
    plot(
        np.argmax(train_y, axis=1),
        train_pred,
        train_prob,
        RESULTS_PLOT,
        CLASSES,
        f"results/{name}/train",
    )

    test_prob = model.predict(test_x)
    test_pred = np.argmax(test_prob, axis=1)
    plot(
        np.argmax(test_y, axis=1),
        test_pred,
        test_prob,
        RESULTS_PLOT,
        CLASSES,
        f"results/{name}/test",
    )


if __name__ == "__main__":
    # train("UNSWNB15")
    train("EdgeIIOT")
