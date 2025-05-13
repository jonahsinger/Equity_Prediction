import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def train_linear_model(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nLinear Regression model trained.")
    # Print feature coefficients
    print("\nFeature Coefficients:")
    if hasattr(X_train, 'columns'): # Check if X_train is a DataFrame
        features = X_train.columns
    else:
        features = [f'feature_{i}' for i in range(X_train.shape[1])]
        
    for feature, coef in zip(features, model.coef_):
        print(f"{feature}: {coef:.6f}")
    return model

def evaluate_linear_model(model, X_train, y_train, X_test, y_test):
    """Evaluates the Linear Regression model and prints metrics."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print("\nLinear Regression Performance Metrics:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Testing MSE: {test_mse:.6f}")
    print(f"Training RMSE: {train_rmse:.6f}")
    print(f"Testing RMSE: {test_rmse:.6f}")
    print(f"Training MAE: {train_mae:.6f}")
    print(f"Testing MAE: {test_mae:.6f}")
    
    # Calculate and print the errors (Residuals for the test set)
    errors = y_test - y_pred_test
    print(f'\nTest Set Errors (Residuals):')
    print(f'Mean Error: {errors.mean():.6f}')
    print(f'Standard Deviation of Errors: {errors.std():.6f}')

    return y_pred_test # Return test predictions for potential further use

def plot_linear_regression(y_test, y_pred_test, filename='linear_regression_results.png'):
    """Plots actual vs predicted values for Linear Regression."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"\nPlot saved as '{filename}'")

if __name__ == "__main__":
    
    # Load the data
    df = pd.read_csv('data.csv')

    # Apply filtering
    lower_bound = -0.15
    upper_bound = 0.15
    original_count = len(df)
    df_filtered = df[(df['y'] > lower_bound) & (df['y'] < upper_bound)]
    filtered_count = len(df_filtered)
    print(f"Original data points: {original_count}")
    print(f"Data points after filtering 'y' between {lower_bound} and {upper_bound}: {filtered_count}")
    print(f"Removed {original_count - filtered_count} data points.")

    # Prepare features and target
    X = df_filtered.drop(['date', 'ticker', 'y'], axis=1)
    y = df_filtered['y']

    # Print data shapes before split
    print("\nData shapes before split (after filtering):")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nData shapes after split:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Train, evaluate, and plot
    model = train_linear_model(X_train, y_train)
    y_pred = evaluate_linear_model(model, X_train, y_train, X_test, y_test)
    plot_linear_regression(y_test, y_pred)
