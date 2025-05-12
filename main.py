import pandas as pd
from train import load_and_prepare_data, split

# Load the data from CSV
df = pd.read_csv('data.csv')
df = load_and_prepare_data("data.csv")

# Display the DataFrame
print("\nDataFrame Shape:", df.shape)
print("\nFirst few rows of the data:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())

X_train, X_test, y_train, y_test = split(df)
X_train = X_train.drop(columns=["ticker"])
X_test = X_test.drop(columns=["ticker"])

print("\n[2] Train/Test Split:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print("\nSample of training features:")
print(X_train.head())
print("\nSample of training labels:")
print(y_train.head())
