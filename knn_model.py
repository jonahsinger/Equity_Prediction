import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def train_knn_model(X_train_scaled, y_train, k=5):
    """Trains a KNN Regressor model."""
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    print(f"\nKNN model (k={k}) trained.")
    return knn

def evaluate_knn_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """Evaluates the KNN model and prints metrics."""
    k_neighbors = model.n_neighbors
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"\nKNN Model (k={k_neighbors}) Performance Metrics:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Testing MSE: {test_mse:.6f}")
    print(f"Training RMSE: {train_rmse:.6f}")
    print(f"Testing RMSE: {test_rmse:.6f}")
    print(f"Training MAE: {train_mae:.6f}")
    print(f"Testing MAE: {test_mae:.6f}")
    print(f"Testing R^2 Score: {test_r2:.6f}")
    
    # Calculate and print the errors (Residuals for the test set)
    errors = y_test - y_pred_test
    print(f'\nTest Set Errors (Residuals):')
    print(f'Mean Error: {errors.mean():.6f}')
    print(f'Standard Deviation of Errors: {errors.std():.6f}')
    
    return y_pred_test # Return test predictions for potential further use

def plot_knn_regression(y_test, y_pred_test, k, filename='knn_regression_results_filtered.png'):
    """Plots actual vs predicted values for KNN Regression."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'KNN Regression (k={k})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"\nPlot saved as '{filename}'")

# --- Main execution block (optional, can be run standalone) ---
if __name__ == "__main__":

    # Load data from data.csv
    data = pd.read_csv('data.csv')

    # --- Filtering Step ---
    lower_bound = -0.15
    upper_bound = 0.15
    original_count = len(data)
    data_filtered = data[(data['y'] > lower_bound) & (data['y'] < upper_bound)]
    filtered_count = len(data_filtered)
    print(f"Original data points: {original_count}")
    print(f"Data points after filtering 'y' between {lower_bound} and {upper_bound}: {filtered_count}")
    print(f"Removed {original_count - filtered_count} data points.")

    # Use the filtered data from now on
    X = data_filtered.drop(['date', 'ticker', 'y'], axis=1)
    y = data_filtered['y']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose k
    k_neighbors = 5 
    
    # Train, evaluate, and plot
    knn_model = train_knn_model(X_train_scaled, y_train, k=k_neighbors)
    y_pred = evaluate_knn_model(knn_model, X_train_scaled, y_train, X_test_scaled, y_test)
    plot_knn_regression(y_test, y_pred, k=k_neighbors)

# Note: You can now analyze the 'knn_regression_results_filtered.png' 
# to choose an optimal k and then re-run a single KNN model 
# with that k for detailed performance metrics if needed. 