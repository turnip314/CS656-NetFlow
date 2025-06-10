import pandas as pd
from sklearn.model_selection import train_test_split

def load_unsw_nb15(
        csv_paths, 
        train_frac=0.8,
        columns=[1,3,4,5,6,7,16,48],
        randomize=True, 
        random_seed=42, 
        max_rows=None):
    """
    Load and split the UNSW-NB15 dataset with optional row limit.

    Parameters:
        csv_paths (list of str): Paths to the four CSV files.
        train_frac (float): Proportion of the data to use for training (0 < train_frac < 1).
        randomize (bool): Whether to shuffle the data before splitting.
        random_seed (int): Seed for reproducibility.
        max_rows (int or None): Maximum number of total rows to load (from all CSVs combined).

    Returns:
        (X_train, X_test, y_train, y_test): Tuple of train/test features and labels.
    """
    # Load and concatenate all CSVs
    dataframes = []
    rows_loaded = 0
    for path in csv_paths:
        if max_rows is not None and rows_loaded >= max_rows:
            break
        remaining_rows = None if max_rows is None else max_rows - rows_loaded
        df = pd.read_csv(path, header = None, low_memory=False, nrows=remaining_rows, usecols=columns)
        dataframes.append(df)
        rows_loaded += len(df)

    full_df = pd.concat(dataframes, ignore_index=True)

    # Optional shuffle
    if randomize:
        full_df = full_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Assume last column is the label
    X = full_df.iloc[:, :-1]
    y = full_df.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_frac, random_state=random_seed if randomize else None, shuffle=randomize
    )

    return X_train, X_test, y_train, y_test
