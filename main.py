import os
import pandas as pd
import time
from src.data_ingestion import load_data, clean_data
from src.data_analysis import summary_statistics, calculate_volatility
from src.data_visualization import plot_stock_prices, plot_model_predictions
from src.predictive_modeling import train_models, evaluate_models
from sklearn.model_selection import train_test_split

def main():
    # Start the timer
    start_time = time.time()

    # File paths
    raw_data_path = 'data/raw/financial_data.csv'
    processed_data_path = 'data/processed/financial_data_cleaned.csv'

    # Step 1: Check if processed data exists
    if os.path.exists(processed_data_path):
        print(f"Loading processed data from {processed_data_path}")
        data = pd.read_csv(processed_data_path)
    else:
        print(f"Processed data not found. Loading raw data from {raw_data_path}")
        data = load_data(raw_data_path)
        cleaned_data = clean_data(data)
        cleaned_data.to_csv(processed_data_path, index=False)
        print(f"Cleaned data saved to {processed_data_path}")
        data = cleaned_data

    # Step 2: Perform basic analysis
    print("Summary Statistics:")
    print(summary_statistics(data))

    print("Volatility of AAPL:")
    print(calculate_volatility(data, 'AAPL'))

    # Step 3: Visualize stock prices
    plot_stock_prices(data, 'AAPL')

    # Step 4: Prepare data for predictive modeling
    X = data.drop(columns=['Date', 'AAPL'])  # Example: other stock data as features
    y = data['AAPL']  # Target variable: predicting AAPL stock prices

    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train multiple models
    models = train_models(X_train, y_train)

    # Step 6: Evaluate models
    results = evaluate_models(models, X_test, y_test)
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Step 7: Plot predictions (example: using Random Forest)
    plot_model_predictions(models['Random Forest'], 'Random Forest', X_test, y_test)
    plot_model_predictions(models['Linear Regression'], 'Linear Regression', X_test, y_test)
    plot_model_predictions(models['Decision Tree'], 'Decision Tree', X_test, y_test)

    # End the timer and calculate the total time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main()




