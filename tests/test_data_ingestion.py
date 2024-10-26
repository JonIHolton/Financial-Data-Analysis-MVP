import unittest
from src.data_ingestion import load_data, clean_data

class TestDataIngestion(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/raw/financial_data.csv')
        self.assertTrue(not data.empty)

    def test_clean_data(self):
        data = load_data('data/raw/financial_data.csv')
        cleaned_data = clean_data(data)
        self.assertFalse(cleaned_data.isnull().values.any())
