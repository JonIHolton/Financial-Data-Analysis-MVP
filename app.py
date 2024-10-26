import streamlit as st
import pandas as pd
from src.data_ingestion import clean_data
from src.predictive_modeling import train_models, evaluate_models
from src.data_visualization import plot_stock_prices, plot_model_predictions
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Financial Data Analysis and Prediction")

# Step 1: File Upload
st.header("Step 1: Upload Financial Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data from uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Raw Data", data)

    # Step 2: Clean Data
    st.header("Step 2: Cleaned Data")
    cleaned_data = clean_data(data)
    st.write("Cleaned Data", cleaned_data)

    # Use a form to let users make their selections and then submit once
    with st.form("stock_and_model_selection"):
        # Step 3: Data Analysis - Stock Selection
        st.header("Step 3: Data Analysis")
        stock_column = st.selectbox("Select Stock for Analysis", cleaned_data.columns[1:])

        # Step 4: Model Selection
        st.header("Step 4: Model Selection")
        available_models = ['Linear Regression', 'Random Forest', 'Decision Tree']
        if 'XGBoost' in available_models:
            available_models.append('XGBoost')
        selected_models = st.multiselect("Select Models for Prediction", available_models, default=['Linear Regression'])

        # Submit button
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        # Plot the stock prices only after the user submits
        plot_stock_prices(cleaned_data, stock_column)

        # Prepare data for predictive modeling
        X = cleaned_data.drop(columns=['Date', stock_column])
        y = cleaned_data[stock_column]

        # Split into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the selected models
        st.header("Step 5: Train Models")
        models = train_models(X_train, y_train, selected_models)
        st.write("Models Trained Successfully")

        # Evaluate models and display metrics
        st.header("Step 6: Model Evaluation")
        results = evaluate_models(models, X_test, y_test)
        for model_name, metrics in results.items():
            st.subheader(f"Model: {model_name}")
            st.write(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
            st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
            st.write(f"R-squared: {metrics['R-squared']:.4f}")

        # Step 7: Plot Model Predictions for all selected models
       # Step 7: Plot Model Predictions for all selected models
        st.header("Step 7: Plot Predictions")
        for model_name in selected_models:
            st.subheader(f"Predictions for {model_name}")
            plot_model_predictions(models[model_name], model_name, X_test, y_test, stock_column)  # Pass stock_column






# import streamlit as st
# import pandas as pd
# from src.data_ingestion import clean_data
# from src.predictive_modeling import train_models, evaluate_models
# from src.data_visualization import plot_stock_prices, plot_model_predictions
# from sklearn.model_selection import train_test_split

# # Title of the app
# st.title("Financial Data Analysis and Prediction")

# # Step 1: File Upload
# st.header("Step 1: Upload Financial Data")
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     # Load data from uploaded file
#     data = pd.read_csv(uploaded_file)
#     st.write("Raw Data", data)

#     # Step 2: Clean Data
#     st.header("Step 2: Cleaned Data")
#     cleaned_data = clean_data(data)
#     st.write("Cleaned Data", cleaned_data)

#     # Step 3: Data Analysis
#     st.header("Step 3: Data Analysis")
#     stock_column = st.selectbox("Select Stock for Analysis", cleaned_data.columns[1:])
#     plot_stock_prices(cleaned_data, stock_column)

#     # Prepare data for predictive modeling
#     X = cleaned_data.drop(columns=['Date', stock_column])
#     y = cleaned_data[stock_column]

#     # Split into train and test datasets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Step 4: Train Models
#     st.header("Step 4: Train Models")
#     models = train_models(X_train, y_train)
#     st.write("Models Trained Successfully")

#     # Step 5: Model Evaluation
#     st.header("Step 5: Model Evaluation")
#     results = evaluate_models(models, X_test, y_test)

#     # Display model performance metrics
#     for model_name, metrics in results.items():
#         st.subheader(f"Model: {model_name}")
#         st.write(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
#         st.write(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
#         st.write(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
#         st.write(f"R-squared: {metrics['R-squared']:.4f}")

#     # Step 6: Plot Model Predictions
#     st.header("Step 6: Plot Predictions")
#     selected_model = st.selectbox("Select Model for Prediction", models.keys())
#     plot_model_predictions(models[selected_model], selected_model, X_test, y_test)
