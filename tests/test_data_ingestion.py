import unittest
import pandas as pd
from src.data_ingestion import load_data, clean_data

class TestDataIngestion(unittest.TestCase):
    def test_load_data(self):
        # Test if the data loads successfully
        data = load_data('data/raw/financial_data.csv')
        self.assertFalse(data.empty, "The data should not be empty after loading.")

    def test_clean_data(self):
        # Test if missing values are handled properly
        raw_data = pd.DataFrame({
            'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'AAPL': [100, None, 105]
        })
        cleaned_data = clean_data(raw_data)
        self.assertFalse(cleaned_data.isnull().values.any(), "The data should not have any missing values.")

