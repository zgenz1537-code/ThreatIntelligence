if True:
    from reset_random import reset_random

    reset_random()
import os

import numpy as np
import pandas as pd

from AOA import AchimedesOptimizer

CLASS = "class"


def save_data(df, cols, save_dir):
    columns = list(df.columns)
    selected_cols = []
    for col in cols:
        selected_cols.append(columns[col])
    print("[INFO] Total Features :: {0}".format(len(df.columns) - 1))
    print("[INFO] Selected Features :: {0}".format(len(selected_cols)))
    with open(os.path.join(save_dir, "selected_features.txt"), "w") as f:
        f.write("\n".join(selected_cols))
    selected_cols.extend([CLASS])
    sp = os.path.join(save_dir, "features_selected.csv")
    print("[INFO] Saving Feature Selected Data To File :: {0}".format(sp))
    df[selected_cols].to_csv(sp, index=False)
    return df[selected_cols]


def select_features(df, name):
    print(f"[INFO] Working on {name} Dataset")
    reset_random()
    sd = f"Data/features/{name}"
    os.makedirs(sd, exist_ok=True)
    df_ = df.copy(deep=True)
    x_, y_ = df.values[:, :-1], df.values[:, -1]
    print("[INFO] Feature Selection Using Star Fish Optimizer")
    solution = AchimedesOptimizer(
        num_agents=1,
        max_iter=25,
        train_data=x_,
        train_label=y_,
        save_conv_graph=True,
        save_dir=sd,
    )
    selected_feats = np.flatnonzero(solution.best_agent)
    return save_data(df_, selected_feats, sd)


if __name__ == "__main__":
    select_features(pd.read_csv("Data/preprocessed/UNSWNB15.csv"), "UNSWNB15")
    select_features(pd.read_csv("Data/preprocessed/EdgeIIOT.csv"), "EdgeIIOT")
