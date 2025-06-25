import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import fbeta_score

from dataloader import load_unsw_nb15

import numpy as np
import time

# 249050

DATA_ROOT = "D:/CS656/Project/data/"
csvs = [
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv"
    ]
csvs = [DATA_ROOT + filename for filename in csvs]

def classify(frac = 0.8, size = 700000, randomize = False):
    t = time.monotonic()
    print(f"Starting test for size {size}.")
    X_train, X_test, y_train, y_test = load_unsw_nb15(csvs, train_frac=frac, randomize=randomize, max_rows=size)
    print(f"Data loaded in time {time.monotonic() - t}")
    t = time.monotonic()
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    print(f"Training completed time {time.monotonic() - t}")

    y_pred = rf.predict(X_test)
    f2_score = fbeta_score(y_test, y_pred, beta=2, average="binary")
    print("F2 score:", f2_score)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print()
    return f2_score

if __name__ == "__main__":
    classify(frac=0.6, size=700000, randomize=True)