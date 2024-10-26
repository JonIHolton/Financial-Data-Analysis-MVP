import pandas as pd

def load_data(file_path):
    """Load financial data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(data):
    """Clean the data by filling missing values."""
    data.fillna(method='ffill', inplace=True)
    return data
