import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from dataloader import load_unsw_nb15

DATA_ROOT = "D:/CS656/Project/data/"
csvs = [
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv"
    ]
csvs = [DATA_ROOT + filename for filename in csvs]

def preprocess_data():
    X_train, X_test, y_train, y_test = load_unsw_nb15(csvs, train_frac=0.75, randomize=False, max_rows=1000)
    return X_train, X_test, y_train, y_test


preprocess_data()
