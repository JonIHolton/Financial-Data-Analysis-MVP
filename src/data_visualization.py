import plotly.graph_objects as go
import pandas as pd

def plot_stock_prices(data, stock_column):
    fig = go.Figure()

    # Add trace with a custom color
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data[stock_column], 
        mode='lines', 
        name=stock_column,
        line=dict(color='blue')  # Set the line color here
    ))

    # Update layout
    fig.update_layout(
        title=f"{stock_column} Stock Prices", 
        xaxis_title='Date', 
        yaxis_title='Price'
    )

    # Show the figure
    fig.show()



import plotly.graph_objects as go

def plot_model_predictions(model, model_name, X_test, y_test, stock_name):
    # Generate predictions
    predictions = model.predict(X_test)

    # Create a Plotly figure for actual vs predicted
    fig = go.Figure()

    # Add actual values trace (set color to blue for actual values)
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))), 
        y=y_test, 
        mode='lines', 
        name='Actual', 
        line=dict(color='blue')
    ))

    # Add predicted values trace (set color to red for predicted values)
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))), 
        y=predictions, 
        mode='lines', 
        name=f'Predicted ({model_name})',
        line=dict(color='red')
    ))

    # Update layout with title and labels, including the stock name
    fig.update_layout(
        title=f"{stock_name} Predictions vs Actual (Model: {model_name})",
        xaxis_title='Test Samples',
        yaxis_title='Stock Price',
        legend_title='Legend'
    )

    # Show the figure
    fig.show()


# def plot_model_predictions(model, model_name, X_test, y_test):
#     # Generate predictions
#     predictions = model.predict(X_test)

#     # Create a Plotly figure for actual vs predicted
#     fig = go.Figure()

#     # Add actual values trace with a custom color
#     fig.add_trace(go.Scatter(
#         x=list(range(len(y_test))), 
#         y=y_test, 
#         mode='lines', 
#         name='Actual',
#         line=dict(color='green')  # Set the actual values color here
#     ))

#     # Add predicted values trace with a custom color
#     fig.add_trace(go.Scatter(
#         x=list(range(len(y_test))), 
#         y=predictions, 
#         mode='lines', 
#         name=f'Predicted ({model_name})',
#         line=dict(color='red')  # Set the predicted values color here
#     ))

#     # Update layout with title and labels
#     fig.update_layout(
#         title=f"Model Predictions vs Actual (Model: {model_name})",
#         xaxis_title='Test Samples',
#         yaxis_title='Stock Price',
#         legend_title='Legend'
#     )

#     # Show the figure
#     fig.show()