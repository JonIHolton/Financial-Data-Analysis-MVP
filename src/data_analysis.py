def summary_statistics(data):
    """Return basic statistics for the financial data."""
    return data.describe()

def calculate_volatility(data, column):
    """Calculate the volatility (standard deviation) of a specific column."""
    return data[column].std()
