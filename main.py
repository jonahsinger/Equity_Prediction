import pandas as pd
from train import load_and_prepare_data, split, preprocess_data, train_model, evaluate_model, plot_results

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

X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

model = train_model(X_train_scaled, y_train)

y_pred = evaluate_model(model, X_test_scaled, y_test)

plot_results(y_test, y_pred)
