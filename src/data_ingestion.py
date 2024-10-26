import pandas as pd

def load_data(file_path):
    """Load financial data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(data):
    # Handle non-numeric values (e.g., 'error') by replacing with NaN
    data.replace('error', pd.NA, inplace=True)
    
    # Forward fill missing values (NaN) using the previous row's value
    data.fillna(method='ffill', inplace=True)
    
    # Backward fill remaining NaN values (if any)
    data.fillna(method='bfill', inplace=True)

    # Optional: Drop rows with remaining NaN values
    # data.dropna(inplace=True)

    # Save the cleaned data to the processed folder
    data.to_csv('data/processed/financial_data_cleaned.csv', index=False)


    return data
