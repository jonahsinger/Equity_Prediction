import pandas as pd
from prepare import load_and_prepare_data, split, preprocess_data
from random_forest import train_rf_model, evaluate_rf, plot_model_vs_actual
from gradient_boost import train_gb_model, evaluate_gb

df = pd.read_csv('data.csv')
df = load_and_prepare_data("data.csv")

print("\nDataFrame Shape:", df.shape)
print("\nFirst few rows of the data:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())

X_train, X_test, y_train, y_test = split(df)

X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = train_rf_model(X_train_scaled, y_train)
rf_pred = evaluate_rf(rf_model, X_test_scaled, y_test)
print("Random Forest evaluation complete.")

# Train Gradient Boosting model
print("\nTraining Gradient Boosting model...")
gb_model = train_gb_model(X_train_scaled, y_train)
gb_pred = evaluate_gb(gb_model, X_test_scaled, y_test)
print("Gradient Boosting evaluation complete.")

models_results = {
    'Random Forest': rf_pred,
    'Gradient Boosting': gb_pred
}

plot_model_vs_actual(y_test, rf_pred, 'Random Forest', 'rf')
plot_model_vs_actual(y_test, gb_pred, 'Gradient Boosting', 'gb')
