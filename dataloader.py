import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
    # Step 1: Separate classes
    df_anomaly = df[df["Label"] == 1]
    df_normal = df[df["Label"] == 0]

    n_anomalies = len(df_anomaly)
    n_normals_needed = int((1 - desired_ratio) / desired_ratio * n_anomalies)

    if n_normals_needed > len(df_normal):
        raise ValueError(f"Not enough normal records to match desired ratio. Needed: {n_normals_needed}, available: {len(df_normal)}")

    df_normal_sampled = df_normal.sample(n=n_normals_needed, random_state=random_state)

    # Step 2: Combine and shuffle
    df_combined = pd.concat([df_anomaly, df_normal_sampled], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_combined

def downsample_preserve_ratio(df, total_size=700000, random_state=1):
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
    value_counts = df["Label"].value_counts(normalize=True)
    ratios = value_counts.to_dict()

    # Sample each class proportionally
    dfs = []
    for label, ratio in ratios.items():
        n_samples = int(total_size * ratio)
        subset = df[df["Label"] == label].sample(n=n_samples, random_state=random_state)
        dfs.append(subset)

    # Combine and shuffle
    result = pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result

def apply_le_transform(df):
    encoder = LabelEncoder()
    df["proto"] = encoder.fit_transform(df["proto"])
    df["state"] = encoder.fit_transform(df["state"])
    return df

def apply_oh_transform(df):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_cols = ["proto", "state"]
    cat_data = df[cat_cols]
    encoded = encoder.fit_transform(cat_data)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    # Step 6: Drop original columns and concatenate
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df

def load_unsw_nb15(
        csv_paths, 
        test_frac=0.2,
        randomize=True, 
        random_seed=1, 
        max_rows=None,
        test_original=False,
        encoding="le"
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

    # Load and concatenate all CSVs
    dataframes = []
    rows_loaded = 0
    for path in csv_paths:
        df = pd.read_csv(path, header = None, low_memory=False, usecols=columns)
        dataframes.append(df)
        rows_loaded += len(df)

    full_df = pd.concat(dataframes, ignore_index=True)
    full_df.columns = columns_to_use
    if encoding == "le":
        full_df = apply_le_transform(full_df)
    elif encoding == "oh":
        full_df = apply_oh_transform(full_df)
    full_df = full_df.apply(pd.to_numeric, errors='coerce')

    full_df = full_df.dropna()

    if test_original:
        return full_df
    
    partial_df = create_netflow_synthetic(full_df, desired_ratio=0.2031, random_state=random_seed)
    partial_df = downsample_preserve_ratio(partial_df, total_size=max_rows, random_state=random_seed)
    #print(partial_df)
    #print(partial_df['Label'].value_counts())
    
    if randomize:
        partial_df = partial_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Assume last column is the label
    X = partial_df.drop(columns=["Label"])
    y = partial_df["Label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=1, stratify=y
    )

    return X_train, X_test, y_train, y_test
