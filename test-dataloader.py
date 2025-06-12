from dataloader import load_unsw_nb15
import pandas as pd

DATA_ROOT = "D:/CS656/Project/data/"
csvs = [
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv"
    ]
csvs = [DATA_ROOT + filename for filename in csvs]


data = load_unsw_nb15(csvs, train_frac=0.99, randomize=False, max_rows=None, test_original=True)
print(len(data[4].unique()))
print(len(data[5].unique()))
print()
#print(sorted(data[1].unique()))
#print(sorted(data[3].unique()))
