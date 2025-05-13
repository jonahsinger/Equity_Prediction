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


def plot_scatter_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))

    plt.scatter(y_test, y_pred, alpha=0.5)

    # ideal line (y = x)
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.title(f'{model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'actual_vs_predicted_{model_name.lower().replace(" ", "_")}.png')
    plt.show()


