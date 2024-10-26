import matplotlib.pyplot as plt

def plot_stock_prices(data, stock_column):
    """Plot stock prices over time."""
    plt.figure(figsize=(10,6))
    plt.plot(data['Date'], data[stock_column])
    plt.title(f'{stock_column} Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
