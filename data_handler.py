if True:
    from reset_random import reset_random

    reset_random()
import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import CLASSES_EDGE_IIOT, CLASSES_UNSWNB15, print_df_to_table

pd.set_option("use_inf_as_na", True)


def show_count(df, class_col, name):
    SORT = {
        "UNSWNB15": CLASSES_UNSWNB15,
        "EdgeIIOT": CLASSES_EDGE_IIOT,
    }
    print("[INFO] Class Distribution")
    cvc = df[class_col].value_counts(sort=False)[SORT[name]]
    sdf = cvc.to_frame()
    sdf.insert(0, "cls", cvc.index)
    sdf.columns = ["Class", "Count"]
    print_df_to_table(sdf)
    return df


def replace_categorical_cols(df):
    for col in df.columns:
        if col == "class":
            continue
        if df[col].dtype != "object":
            continue
        print("[INFO] Replacing Categorical Values in Column :: {0}".format(col))
        repd = {v: k + 1 for k, v in enumerate(sorted(df[col].unique()))}
        df[col].replace(repd, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_unswnb15():
    print("[INFO] Working on UNSWNB15 Dataset")
    dp = glob.glob(os.path.join("Data/source/UNSWNB15", "*.csv"))
    dfs = []
    for f in dp:
        print("[INFO] Loading Data From :: {0}".format(f))
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    df = pd.concat(dfs)
    df.columns = list(map(str.strip, list(df.columns)))[:-1] + ["class"]
    rp_dict = {
        "WebAttack BruteForce": ["Web Attack � Brute Force"],
        "WebAttack XSS": ["Web Attack � XSS"],
    }
    for p in rp_dict:
        for k in rp_dict[p]:
            df["class"].replace({k: p}, inplace=True)
    df.dropna(inplace=True)
    df.columns = df.columns.tolist()[:-1] + ["class"]
    df["class"].replace(
        {
            "BENIGN": "Benign",
            "DoS slowloris": "DoS Slowloris",
        },
        inplace=True,
    )
    dfs = []
    for c in CLASSES_UNSWNB15:
        sdf = df[df["class"] == c].head(3000)
        dfs.append(sdf)
    df = pd.concat(dfs)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except:
            pass
    df = replace_categorical_cols(df)
    x, y = df.values[:, :-1], df.values[:, -1]
    print("[INFO] Normalizing Data")
    mm = StandardScaler()
    x = mm.fit_transform(x, y)
    df = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], axis=1), columns=df.columns)
    show_count(df, "class", "UNSWNB15")
    df["class"].replace({c: i for i, c in enumerate(CLASSES_UNSWNB15)}, inplace=True)
    os.makedirs("Data/preprocessed", exist_ok=True)
    dp = "Data/preprocessed/UNSWNB15.csv"
    print("[INFO] Data Shape After Preprocessed :: {0}".format(df.shape))
    print("[INFO] Saving Preprocessed Data :: {0}".format(dp))
    df.to_csv(dp, index=False)


def load_edge_iiot():
    print("[INFO] Working on EdgeIIOT Dataset")
    df = pd.read_csv("Data/source/DNN-EdgeIIOT-dataset.csv", low_memory=False)
    df.dropna(inplace=True)
    df.drop(["frame.time", "Attack_label"], axis=1, inplace=True)
    df.columns = df.columns.tolist()[:-1] + ["class"]
    df["class"].replace(
        {
            "DDoS_UDP": "DDoS-UDP",
            "DDoS_ICMP": "DDoS-ICMP",
            "SQL_injection": "SQL injection",
            "DDoS_TCP": "DDoS-TCP",
            "DDoS_HTTP": "DDoS-HTTP",
        },
        inplace=True,
    )
    dfs = []
    for c in CLASSES_EDGE_IIOT:
        sdf = df[df["class"] == c].head(5000)
        dfs.append(sdf)
    df = pd.concat(dfs)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except:
            pass
    df = replace_categorical_cols(df)
    x, y = df.values[:, :-1], df.values[:, -1]
    print("[INFO] Normalizing Data")
    mm = StandardScaler()
    x = mm.fit_transform(x, y)
    df = pd.DataFrame(np.concatenate([x, y.reshape(-1, 1)], axis=1), columns=df.columns)
    show_count(df, "class", "EdgeIIOT")
    df["class"].replace({c: i for i, c in enumerate(CLASSES_EDGE_IIOT)}, inplace=True)
    os.makedirs("Data/preprocessed", exist_ok=True)
    dp = "Data/preprocessed/EdgeIIOT.csv"
    print("[INFO] Data Shape After Preprocessed :: {0}".format(df.shape))
    print("[INFO] Saving Preprocessed Data :: {0}".format(dp))
    df.to_csv(dp, index=False)


if __name__ == "__main__":
    # load_unswnb15()
    load_edge_iiot()
