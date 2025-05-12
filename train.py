import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt



RANDOM_SEED = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10
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

def train_model(X_train, y_train, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return y_pred


def plot_results(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Returns', marker='o', alpha=0.6)
    plt.plot(y_test.index, y_pred, label='Predicted Returns', marker='x', alpha=0.6)
    plt.title('Weekly Stock Return Prediction')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('return_predictions.png')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.title('Actual vs Predicted Returns')
    plt.xlabel('Actual Return')
    plt.ylabel('Predicted Return')
    plt.grid(True, alpha=0.3)
    plt.savefig('actual_vs_predicted.png')
    plt.close()