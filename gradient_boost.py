import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

N_ESTIMATORS = 200  # Default number of estimators for Gradient Boosting
MAX_DEPTH = 5      # Default max depth (typically lower than RF)
LEARNING_RATE = 0.1 
RANDOM_SEED = 42
MODEL_FILENAME = "gradient_boosting_model.joblib"

def train_gb_model(X_train, y_train, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE):
    print(f"\nTraining Gradient Boosting model with {n_estimators} trees, max_depth={max_depth}, learning_rate={learning_rate}")
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

def evaluate_gb(model, X_test, y_test):
    print("\nGradient Boosting Model Evaluation:")
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

def plot_model_vs_actual(y_test, y_pred, label, filename_suffix):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Returns', color='black', linestyle='--', marker='o', alpha=0.5)
    plt.plot(y_test.index, y_pred, label=f'{label} Prediction', marker='x', alpha=0.6)

    plt.title(f'{label} Weekly Stock Return Prediction')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'return_predictions_{filename_suffix}.png')
    plt.show()
