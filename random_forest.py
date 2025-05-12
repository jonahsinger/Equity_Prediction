import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_SEED = 42
def train_rf_model(X_train, y_train, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH):
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


def evaluate_rf(model, X_test, y_test):
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

