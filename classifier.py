from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import fbeta_score, roc_auc_score

from dataloader import load_unsw_nb15
import random

from pathlib import Path
this_dir = Path(__file__).parent

import numpy as np
import time

DATA_ROOT = this_dir/"data"
csvs = [
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv"
    ]
csvs = [DATA_ROOT/filename for filename in csvs]

def get_classifier(name="rf", metadata=None):
    if name == "rf":
        return RandomForestClassifier()
    elif name == "sgd":
        return SGDClassifier()
    elif name == "dt":
        return DecisionTreeClassifier()
    elif name == "svm":
        return LinearSVC()
    elif name == "gnb":
        return GaussianNB()
    elif name == "knn":
        return KNeighborsClassifier()
    elif name == "ab":
        return AdaBoostClassifier()
    elif name == "ab-custom":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=metadata), n_estimators=200, learning_rate=0.5)
    raise Exception(f"Invalid classifier: {name}.")

def load_data(
        frac = 0.2, 
        size = 700000, 
        randomize = False, 
        encoding="le", 
        anomalous_ratio = 0.2031, 
        downscale=True,
        idx = (0,1,2,3)
    ):
    if idx != (0,1,2,3):
        import dataloader
        dataloader._cached_df = None
    if randomize:
        return load_unsw_nb15(
            [csvs[i] for i in idx], 
            test_frac=frac, 
            randomize=randomize, 
            random_seed=random.randint(0,99999999), 
            max_rows=size, 
            encoding=encoding,
            do_downscale=downscale,
            anomalous_ratio=anomalous_ratio
        )
    else:
        return load_unsw_nb15(
            [csvs[i] for i in idx], test_frac=frac, 
            randomize=randomize, 
            max_rows=size, 
            encoding=encoding,
            do_downscale=downscale,
            anomalous_ratio=anomalous_ratio
        )

def classify(cf_name, X_train, y_train, metadata=None):
    cf = get_classifier(cf_name, metadata)
    t = time.monotonic()
    cf.fit(X_train, y_train)
    return cf, time.monotonic() - t

def test(cf, X_test, y_test, metric="f2"):
    y_pred = cf.predict(X_test)
    if metric == "acc":
        return accuracy_score(y_test, y_pred)
    elif metric == "f2":
        return fbeta_score(y_test, y_pred, beta=2, average="binary")
    elif metric == "auc":
        return roc_auc_score(y_test, y_pred)
    return -1

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(encoding="oh")
    cf = classify("rf", X_train, y_train)
    print(test(cf, X_test, y_test))