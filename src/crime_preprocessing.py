from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """
    Load the crime dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    print(f"Loading data from {file_path}...")
    # Read CSV with low_memory=False to avoid DtypeWarning for mixed types
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def clean_data(df):
    """
    Clean and preprocess the crime dataset for machine learning.

    Args:
        df (pd.DataFrame): Raw crime data

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    print("Starting data cleaning process...")

    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()

    # 1. Handle missing values
    print(f"Missing values before cleaning:\n{cleaned_df.isnull().sum().sum()}")

    # For columns with significant missing values, decide whether to drop or impute
    # Drop rows with missing values in critical columns
    critical_columns = ["Date", "Primary Type", "Location", "Arrest"]
    cleaned_df = cleaned_df.dropna(subset=critical_columns)

    # 2. Convert date columns to datetime
    if "Date" in cleaned_df.columns:
        cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"])
        # Extract useful time components
        cleaned_df["Year"] = cleaned_df["Date"].dt.year
        cleaned_df["Month"] = cleaned_df["Date"].dt.month
        cleaned_df["Day"] = cleaned_df["Date"].dt.day
        cleaned_df["DayOfWeek"] = cleaned_df["Date"].dt.dayofweek
        cleaned_df["Hour"] = cleaned_df["Date"].dt.hour

    # 3. Handle categorical variables
    # Identify categorical columns (excluding dates and IDs)
    categorical_columns = cleaned_df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_columns:
        # Replace rare categories with 'Other'
        if col in ["Primary Type", "Location Description"]:
            value_counts = cleaned_df[col].value_counts()
            # Categories that appear less than 1% of the time
            mask = value_counts / len(cleaned_df) < 0.01
            rare_categories = value_counts[mask].index
            cleaned_df.loc[cleaned_df[col].isin(rare_categories), col] = "Other"

    # 4. Handle boolean columns
    bool_columns = ["Arrest", "Domestic"]
    for col in bool_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(bool)

    # 5. Handle geographic data
    # Process latitude and longitude if available
    geo_columns = ["Latitude", "Longitude"]
    for col in geo_columns:
        if col in cleaned_df.columns:
            # Replace invalid coordinates with NaN
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    # If both lat and long exist, create a feature for coordinate validity
    if all(col in cleaned_df.columns for col in geo_columns):
        cleaned_df["HasValidCoordinates"] = cleaned_df[geo_columns].notna().all(axis=1)

    # 6. Feature engineering
    # Create a feature for crime count by location
    if "Location Description" in cleaned_df.columns:
        location_counts = cleaned_df["Location Description"].value_counts()
        cleaned_df["LocationFrequency"] = cleaned_df["Location Description"].map(
            location_counts
        )

    # Create time-based features
    if "Date" in cleaned_df.columns:
        # Is weekend
        cleaned_df["IsWeekend"] = cleaned_df["DayOfWeek"].isin([5, 6])
        # Time of day category
        cleaned_df["TimeOfDay"] = pd.cut(
            cleaned_df["Hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True,
        )

    # 7. Remove redundant or unhelpful columns
    columns_to_drop = [
        "ID",  # Unique identifier not useful for prediction
        "Case Number",  # Another ID column
        "IUCR",  # Can use Primary Type instead
        "X Coordinate",
        "Y Coordinate",  # If we have Lat/Long, these are redundant
        "Updated On",  # Not relevant for prediction
        # Add other columns that aren't useful for your specific ML task
    ]
    # Only drop columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in cleaned_df.columns]
    cleaned_df = cleaned_df.drop(columns=columns_to_drop)

    # 8. Check for and handle outliers in numerical columns
    numerical_columns = cleaned_df.select_dtypes(include=["int64", "float64"]).columns
    for col in numerical_columns:
        # Skip ID or categorical columns that might be numeric
        if col.endswith("ID") or col in ["Year", "Month", "Day", "Hour"]:
            continue

        # Calculate IQR and identify outliers
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers (Winsorization)
        cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)

    print(f"Data cleaning complete. New shape: {cleaned_df.shape}")
    print(f"Missing values after cleaning:\n{cleaned_df.isnull().sum().sum()}")

    return cleaned_df


def analyze_data(df):
    """
    Perform exploratory data analysis on the crime dataset.

    Args:
        df (pd.DataFrame): DataFrame to analyze

    Returns:
        dict: Dictionary with analysis results
    """
    print("Performing exploratory data analysis...")
    analysis = {}

    # Basic dataframe information
    analysis["shape"] = df.shape
    analysis["dtypes"] = df.dtypes
    analysis["missing_values"] = df.isnull().sum()

    # Crime type distribution
    if "Primary Type" in df.columns:
        analysis["crime_distribution"] = df["Primary Type"].value_counts()

    # Temporal analysis
    if "Year" in df.columns and "Month" in df.columns:
        analysis["crimes_by_year"] = df["Year"].value_counts().sort_index()
        analysis["crimes_by_month"] = df["Month"].value_counts().sort_index()

    # Location analysis
    if "Location Description" in df.columns:
        analysis["top_locations"] = df["Location Description"].value_counts().head(10)

    # Arrest rate
    if "Arrest" in df.columns:
        analysis["arrest_rate"] = df["Arrest"].mean()

    print("Exploratory data analysis complete!")
    return analysis


def process_features(
    df, categorical_features, numerical_features, train=True, scaler=None
):
    """
    Helper function to process features on a dataframe.

    Args:
        df (pd.DataFrame): DataFrame to process
        categorical_features (list): List of categorical column names
        numerical_features (list): List of numerical column names
        train (bool): Whether this is the training set (to fit transformers)
        scaler (StandardScaler, optional): Pre-fitted scaler for test data

    Returns:
        tuple: (processed_df, scaler) - DataFrame and fitted scaler (if train=True)
    """
    from sklearn.preprocessing import StandardScaler
    import gc

    # Handle categorical variables more efficiently
    # Process one categorical column at a time to reduce memory usage
    categorical_columns = [col for col in categorical_features if col in df.columns]

    # List to keep track of created dummy columns
    all_dummies = []

    for col in categorical_columns:
        print(f"  Encoding column: {col}")
        # Get dummies for this column only
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dummy_na=False)

        # Append to list of all dummies
        all_dummies.append(dummies)

        # Remove original column to save memory
        df = df.drop(columns=[col])

        # Force garbage collection
        gc.collect()

    # Now concatenate all dummy variables with the main dataframe
    if all_dummies:
        # Concatenate all dummy dataframes
        dummy_df = pd.concat(all_dummies, axis=1)

        # Concatenate with original dataframe
        df = pd.concat([df, dummy_df], axis=1)

        # Clear temporary data to free memory
        del all_dummies, dummy_df
        gc.collect()

    # Scale numerical features
    if numerical_features and any(col in df.columns for col in numerical_features):
        # Only use columns that exist in the dataframe
        num_cols = [col for col in numerical_features if col in df.columns]

        if num_cols:
            if train:
                # Create and fit a new scaler on training data
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df[num_cols])
            else:
                # Use the pre-fitted scaler passed as parameter
                if scaler is None:
                    raise ValueError("Scaler must be provided when train=False")
                scaled_features = scaler.transform(df[num_cols])

            # Replace original values with scaled values
            for i, col in enumerate(num_cols):
                df[col] = scaled_features[:, i]

            # Clear memory
            del scaled_features
            gc.collect()

    if train:
        return df, scaler
    else:
        return df


def prepare_for_ml(df, target_column=None, test_size=0.2, random_state=42):
    """
    Prepare the cleaned data for machine learning by encoding categorical features,
    scaling numerical features, and splitting into train/test sets.

    Args:
        df (pd.DataFrame): Cleaned DataFrame
        target_column (str): Column name to use as target variable
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing processed data ready for ML
    """
    from sklearn.model_selection import train_test_split

    print("Preparing data for machine learning...")
    ml_ready_data = {}

    # Create a copy to avoid modifying the input
    processed_df = df.copy()

    # 1. Identify feature types
    categorical_features = processed_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    bool_features = processed_df.select_dtypes(include=["bool"]).columns.tolist()
    numerical_features = processed_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Remove target from features if specified
    if target_column:
        if target_column in categorical_features:
            categorical_features.remove(target_column)
        elif target_column in numerical_features:
            numerical_features.remove(target_column)
        elif target_column in bool_features:
            bool_features.remove(target_column)

    # Store feature names for later reference
    ml_ready_data["feature_names"] = {
        "categorical": categorical_features,
        "numerical": numerical_features,
        "boolean": bool_features,
    }

    # 2. Handle remaining missing values
    # For numerical features: impute with median
    for col in numerical_features:
        processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    # For categorical features: impute with mode
    for col in categorical_features:
        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])

    # 3. Encode categorical variables
    # One-hot encode categorical features
    processed_df = pd.get_dummies(
        processed_df, columns=categorical_features, drop_first=True, dummy_na=False
    )

    # 4. Scale numerical features
    scaler = StandardScaler()
    if numerical_features:
        scaled_features = scaler.fit_transform(processed_df[numerical_features])
        for i, col in enumerate(numerical_features):
            processed_df[col] = scaled_features[:, i]

    ml_ready_data["processed_df"] = processed_df
    ml_ready_data["scaler"] = scaler

    # 5. Split into train and test sets if target is specified
    if target_column:
        if target_column in processed_df.columns:
            X = processed_df.drop(columns=[target_column])
            y = processed_df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            ml_ready_data["X_train"] = X_train
            ml_ready_data["X_test"] = X_test
            ml_ready_data["y_train"] = y_train
            ml_ready_data["y_test"] = y_test

    print("Data is ready for machine learning!")
    return ml_ready_data
