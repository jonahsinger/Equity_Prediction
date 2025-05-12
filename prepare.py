import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt



TEST_SIZE = 0.2

def load_and_prepare_data(file_path="data.csv"):
    # Load the data
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values detected:\n{missing[missing > 0]}")
        print("Dropping rows with missing values...")
        df = df.dropna()
    
    return df


def split(df, test_size=TEST_SIZE):
    df = df.sort_index()
    
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    split_date = df.index[split_idx]

    print(f"\n[!] Train/Test split at index {split_idx}, date: {split_date.date()}")


    X_train = train.drop('y', axis=1)
    y_train = train['y']
    X_test = test.drop('y', axis=1)
    y_test = test['y']
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    X_train = X_train.drop(columns=["ticker"])
    X_test = X_test.drop(columns=["ticker"])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler