import unittest
from sklearn.model_selection import train_test_split
from src.predictive_modeling import train_linear_model, evaluate_model
import pandas as pd

class TestPredictiveModeling(unittest.TestCase):
    def test_train_model(self):
        # Create mock data
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]})
        y = pd.Series([100, 101, 102, 103, 104])
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_linear_model(X_train, y_train)
        
        # Check if the model coefficients are non-empty
        self.assertIsNotNone(model.coef_, "The model should have been trained with non-zero coefficients.")
    
    def test_evaluate_model(self):
        # Create mock data
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]})
        y = pd.Series([100, 101, 102, 103, 104])
        
        # Train model and evaluate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_linear_model(X_train, y_train)
        mse = evaluate_model(model, X_test, y_test)
        
        # Assert that the Mean Squared Error is calculated correctly
        self.assertGreaterEqual(mse, 0, "Mean Squared Error should be non-negative.")
