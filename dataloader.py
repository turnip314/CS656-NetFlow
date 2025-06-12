import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

column_map = {
        "srcip": 0,
        "sport": 1,
        "dstip": 2,
        "dsport": 3,
        "proto": 4,
        "state": 5,
        "dur": 6,
        "sbytes": 7,
        "spkts": 16,
        "stime": 28,
        "ltime": 29,
        "Label": 48
}
columns_to_use = [
    "sport",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "spkts",
    "Label",
]

def create_netflow_synthetic(df, desired_ratio=0.2, random_state=1):
    """
    Create a NetFlow-style dataset from UNSW-NB15 by:
    - Keeping only NetFlow-compatible features
    - Combining all anomalies with a sampled subset of normal traffic

    Parameters:
        df (pd.DataFrame): Original UNSW-NB15 data
        desired_ratio (float): Target anomaly ratio (e.g., 0.2 for 20%)
        random_state (int): Seed for reproducibility

    Returns:
        pd.DataFrame: Synthesized labeled NetFlow-like dataset
    """
    # Step 1: Keep only NetFlow-compatible features + label
    df = df[[column_map[i] for i in columns_to_use]]

    # Step 2: Separate classes
    df_anomaly = df[df[column_map["Label"]] == 1]
    df_normal = df[df[column_map["Label"]] == 0]

    n_anomalies = len(df_anomaly)
    n_normals_needed = int((1 - desired_ratio) / desired_ratio * n_anomalies)

    if n_normals_needed > len(df_normal):
        raise ValueError(f"Not enough normal records to match desired ratio. Needed: {n_normals_needed}, available: {len(df_normal)}")

    df_normal_sampled = df_normal.sample(n=n_normals_needed, random_state=random_state)

    # Step 3: Combine and shuffle
    df_combined = pd.concat([df_anomaly, df_normal_sampled], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_combined

def downsample_preserve_ratio(df, total_size=700_000, random_state=1):
    """
    Downsample a DataFrame to a fixed number of rows while preserving class ratio.

    Parameters:
        df (pd.DataFrame): Input dataset
        label_column (str): Name of the label column
        total_size (int): Desired total number of rows
        random_state (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Downsampled dataset with same class ratio
    """
    # Get the class distribution
    value_counts = df[column_map["Label"]].value_counts(normalize=True)
    ratios = value_counts.to_dict()

    # Sample each class proportionally
    dfs = []
    for label, ratio in ratios.items():
        n_samples = int(total_size * ratio)
        subset = df[df[column_map["Label"]] == label].sample(n=n_samples, random_state=random_state)
        dfs.append(subset)

    # Combine and shuffle
    result = pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result

def load_unsw_nb15(
        csv_paths, 
        train_frac=0.8,
        randomize=True, 
        random_seed=1, 
        max_rows=None,
        test_original=False
):
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
    
    columns = [column_map[col] for col in columns_to_use]

    # For encoding categorical data
    encoder = LabelEncoder()

    # Load and concatenate all CSVs
    dataframes = []
    rows_loaded = 0
    for path in csv_paths:
        df = pd.read_csv(path, header = None, low_memory=False, usecols=columns)
        dataframes.append(df)
        rows_loaded += len(df)

    full_df = pd.concat(dataframes, ignore_index=True)
    full_df[4] = encoder.fit_transform(full_df[4])
    full_df[5] = encoder.fit_transform(full_df[5])
    full_df = full_df.apply(pd.to_numeric, errors='coerce')

    full_df = full_df.dropna()

    if test_original:
        return full_df
    
    print(len(full_df))
    partial_df = create_netflow_synthetic(full_df, desired_ratio=0.2, random_state=random_seed)
    print(len(partial_df))
    print(partial_df[column_map["Label"]].value_counts(normalize=True))
    partial_df = downsample_preserve_ratio(partial_df, total_size=max_rows, random_state=random_seed)
    
    # Optional shuffle
    if randomize:
        partial_df = partial_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(len(partial_df))
    print(partial_df[column_map["Label"]].value_counts(normalize=True))

    # Assume last column is the label
    X = partial_df.iloc[:, :-1]
    y = partial_df.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_frac, random_state=1, stratify=y
    )

    return X_train, X_test, y_train, y_test
