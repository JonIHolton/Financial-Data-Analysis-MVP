import unittest
from src.data_analysis import summary_statistics, calculate_volatility
import pandas as pd

class TestDataAnalysis(unittest.TestCase):
    def test_summary_statistics(self):
        # Test if summary statistics are generated correctly
        data = pd.DataFrame({'AAPL': [100, 102, 104, 98, 105]})
        stats = summary_statistics(data)
        self.assertEqual(stats.loc['mean']['AAPL'], 101.8, "Mean should be correct.")
    
    def test_calculate_volatility(self):
        # Test if volatility is calculated correctly
        data = pd.DataFrame({'AAPL': [100, 102, 104, 98, 105]})
        volatility = calculate_volatility(data, 'AAPL')
        self.assertGreater(volatility, 0, "Volatility should be a positive number.")
