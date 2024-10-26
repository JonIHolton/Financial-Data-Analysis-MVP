from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

try:
    from xgboost import XGBRegressor
    xgboost_installed = True
except ImportError:
    xgboost_installed = False

# Train and return models
def train_models(X_train, y_train, selected_models):
    available_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
    }

    # Include XGBoost if available and selected
    if xgboost_installed:
        available_models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)

    models = {}
    for model_name in selected_models:
        if model_name in available_models:
            model = available_models[model_name]
            model.fit(X_train, y_train)
            models[model_name] = model
    
    return models

# Evaluate models
def evaluate_models(models, X_test, y_test):
    evaluation_results = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        evaluation_results[name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R-squared': r2
        }
    return evaluation_results


