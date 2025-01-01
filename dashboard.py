# Importing necessary libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
from plotly import graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder 
import plotly.figure_factory as ff
import seaborn as sns
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Ignore warnings during execution
warnings.filterwarnings('ignore')  # Ignore any warnings during execution

# Setting the page configuration for the Streamlit app
st.set_page_config(page_title="Superstore EDA", page_icon=":bar_chart:", layout="wide"
)

# Custom CSS styling for the page to enhance the look and feel of the app
st.markdown("""
    <style>
        /* General Layout Enhancements */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        
        .big-font {
            font-size: 48px !important;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            padding: 20px;
            margin-bottom: 30px;
            letter-spacing: 2px;
        }
        
        /* Header Styling */
        h1, h2, h3 {
            color: #333;
            font-family: 'Arial', sans-serif;
            font-weight: 600;
        }
        
        /* Styling for Buttons */
        .stButton>button {
            background-color: #FF6347;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            transition: background-color 0.3s ease, transform 0.3s ease;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
        }
        
        .stButton>button:hover {
            background-color: #e53e32;
            transform: scale(1.05);
        }
        
        /* Streamlit Sidebar - Custom Styling */
        .css-1d391kg {
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }

        /* Sidebar Section Header */
        .css-11kwzxw {
            color: #FF6347;
            font-weight: bold;
            font-size: 20px;
        }

        /* Title for Interactive Filters */
        .css-1pavq6g {
            font-size: 18px;
            font-weight: 500;
            color: #4CAF50;
        }
        
        /* Tables and DataFrames Styling */
        .dataframe {
            margin: 0 auto;
            border-collapse: collapse;
            width: 100%;
        }
        
        .dataframe th, .dataframe td {
            padding: 10px;
            text-align: center;
            border: 1px solid #ddd;
            background-color: #fff;
        }
        
        .dataframe th {
            background-color: #4CAF50;
            color: white;
        }
        
        /* Hover effect for tables */
        .dataframe tbody tr:hover {
            background-color: #f1f1f1;
            cursor: pointer;
        }
        
        /* Streamlit Expander Styling */
        .st-expanderHeader {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        
        /* Plotly Chart Area */
        .plotly-graph-div {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }

        /* Animation on Hover for Plots */
        .plotly-graph-div:hover {
            transform: scale(1.02);
            transition: transform 0.3s ease;
        }

        /* Sidebar - Custom Scrollbar */
        .css-1dp5vc8 {
            background-color: #f1f1f1;
            border-radius: 8px;
        }

        /* Sidebar Widget Styling */
        .stCheckbox, .stRadio {
            font-size: 16px;
            color: #4CAF50;
            font-weight: 500;
        }

        /* Streamlit Widgets - Custom Styling */
        .stSlider {
            font-size: 16px;
            color: #4CAF50;
        }

    </style>
    """, unsafe_allow_html=True)

# Set the main header of the app
st.title(":bar_chart: Sample Superstore EDA")
st.write("An interactive app for exploratory data analysis and forecasting.")

# Add a section header for file handling
st.markdown("<h1 style='font-size:50px;'>FILE HANDLING</h1>", unsafe_allow_html=True)
st.header("1. Upload Your Dataset")

# File uploader widget to allow user to upload a file
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "txt", "xlsx", "xls"])

# Function to load the dataset from the uploaded file (with caching to avoid reloading every time)
@st.cache_data
def load_data(file):
    try:
        # Read the file into a pandas DataFrame
        df = pd.read_excel(file)  # You can switch this to pd.read_csv(file) if uploading a CSV
        return df
    except Exception as e:
        # If an error occurs, display an error message
        st.error(f"Error reading the file: {e}")
        st.stop()

# Load the data if file is uploaded, otherwise load a default file from the local machine
if fl is not None:
    df = load_data(fl)  # Load the dataset from uploaded file
else:
    default_path = r"C:\\Users\\bhoom\\OneDrive\\Documents\\opencv\\myenv\\Superstore.csv"  # Local path for default file
    if os.path.exists(default_path):
        df = load_data(default_path)  # Load the default dataset
    else:
        st.error("Default file not found. Please upload a file.")  # Show error if default file is missing
        st.stop()  # Stop further execution if the file cannot be loaded


# --- Helper Functions ---

# Function to handle missing values in the dataset
def handle_missing_values(df, missing_action):
    # Drop rows with missing values
    if missing_action == "Drop Rows with Missing Values":
        df = df.dropna()
    # Fill missing values with mean for numeric columns 
    elif missing_action == "Fill Missing Values with Mean (Numeric Columns)":
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns  # Select numeric columns
        for col in numeric_columns:
            df[col].fillna(df[col].mean(), inplace=True)  # Fill missing values with mean
    # Fill missing values with median for numeric columns
    elif missing_action == "Fill Missing Values with Median (Numeric Columns)":
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns  # Select numeric columns
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)  # Fill missing values with median
    # Fill missing values with mode for categorical columns
    elif missing_action == "Fill Missing Values with Mode (Categorical Columns)":
        categorical_columns = df.select_dtypes(include=["object"]).columns  # Select categorical columns
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)  # Fill missing values with mode
    # Fill missing values with a custom value
    elif missing_action == "Fill Missing Values with Custom Value":
        custom_value = st.text_input("Enter a custom value to fill missing values:", "Unknown")  # Prompt for custom value
        df.fillna(custom_value, inplace=True)  # Fill missing values with the custom value
    return df  # Return the cleaned dataframe

# Function to filter data based on user-selected location (Region, State, City)
def filter_by_location(df, region, state, city):
    # Handle the different combinations of region, state, and city filtering
    if not region and not state and not city:
        return df  # Return the entire dataset if no location filters are applied
    elif not state and not city:
        return df[df["Region"].isin(region)]  # Filter by region
    elif not region and not city:
        return df[df["State"].isin(state)]  # Filter by state
    elif state and city:
        return df[df["State"].isin(state) & df["City"].isin(city)]  # Filter by state and city
    elif region and city:
        return df[df["Region"].isin(region) & df["City"].isin(city)]  # Filter by region and city
    elif region and state:
        return df[df["Region"].isin(region) & df["State"].isin(state)]  # Filter by region and state
    elif city:
        return df[df["City"].isin(city)]  # Filter by city
    else:
        return df[df["Region"].isin(region) & df["State"].isin(state) & df["City"].isin(city)]  # Filter by all locations

# Function to plot sales by category using a bar chart
def plot_category_sales(df):
    category_df = df.groupby("Category", as_index=False)["Sales"].sum()  # Group data by category and sum sales
    fig = px.bar(category_df, x="Category", y="Sales", text=category_df["Sales"].map("${:,.2f}".format))  # Create bar chart
    return fig  # Return the plot

# Function to plot sales by region using a pie chart
def plot_region_sales(df):
    fig = px.pie(df, values="Sales", names="Region", hole=0.5)  # Create pie chart with a hole
    return fig  # Return the plot

# Function to plot time-series sales data
def plot_time_series_sales(df):
    df["month_year"] = df["Order Date"].dt.to_period("M")  # Extract month-year from order date
    linechart = df.groupby(df["month_year"].dt.strftime("%Y-%b"))["Sales"].sum().reset_index()  # Group by month-year and sum sales
    fig = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"})  # Create line chart
    # Add trendline to the plot    
    fig.update_traces(mode='lines+markers', line=dict(shape='linear'))
    return fig  # Return the plot

# Function to plot sales in a treemap
def plot_sales_treemap(df):
    fig = px.treemap(df, path=["Region", "Category", "Sub-Category"], values="Sales", color="Sub-Category")  # Create treemap
    return fig  # Return the plot

# Function to plot a scatter plot for sales vs. profit
def plot_scatter_sales_profit(df):
    fig = px.scatter(df, x="Sales", y="Profit", size="Quantity", color="Category",
                     hover_data=["Region", "State", "City", "Sub-Category", "Order ID", "Sales", "Profit", "Quantity"])  # Create scatter plot
    return fig  # Return the plot

# --- Sales Forecasting using ARIMA Model ---
def forecast_sales(data, forecast_period=12):
    # Ensure 'Order Date' is in datetime format
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    # Group by 'Order Date' and sum sales
    sales_data = data.groupby('Order Date')['Sales'].sum()  
    # Fit ARIMA model to forecast future sales
    model = ARIMA(sales_data, order=(5, 1, 0))  # ARIMA parameters (p, d, q)
    model_fit = model.fit()
    # Forecast the sales for the next forecast_period months
    forecast = model_fit.forecast(steps=forecast_period)
    # Generate future dates for the forecast period
    last_date = sales_data.index[-1]
    future_dates = pd.date_range(last_date, periods=forecast_period + 1, freq='M')[1:]
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Sales': forecast
    })
    # Plot the forecast
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=sales_data.index, y=sales_data.values, mode='lines', name='Historical Sales'))
    forecast_fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Sales'], mode='lines', name='Forecasted Sales', line=dict(dash='dash')))
    forecast_fig.update_layout(title="Sales Forecast", xaxis_title="Date", yaxis_title="Sales")
    st.plotly_chart(forecast_fig)  # Display the plot

# Function to generate visualizations with error handling
def create_charts(data):
    try:
        # Create columns for charts
        col1, col2, col3 = st.columns(3)
        # Sales by Category (Bar chart)
        with col1:
            if 'Category' in data.columns and 'Sales' in data.columns:
                fig = px.bar(data, x="Category", y="Sales", title="Sales by Category")  # Sales by Category chart
                st.plotly_chart(fig, use_container_width=True)  # Display the chart
            else:
                st.warning("Required columns for 'Sales by Category' chart are missing.")
        # Sales by Region (Bar chart)
        with col2:
            if 'Region' in data.columns and 'Sales' in data.columns:
                fig2 = px.bar(data, x="Region", y="Sales", title="Sales by Region")  # Sales by Region chart
                st.plotly_chart(fig2, use_container_width=True)  # Display the chart
            else:
                st.warning("Required columns for 'Sales by Region' chart are missing.")
        # Profit by Category (Bar chart)
        with col3:
            if 'Category' in data.columns and 'Profit' in data.columns:
                fig3 = px.bar(data, x="Category", y="Profit", title="Profit by Category")  # Profit by Category chart
                st.plotly_chart(fig3, use_container_width=True)  # Display the chart
            else:
                st.warning("Required columns for 'Profit by Category' chart are missing.")
    except Exception as e:
        st.error(f"An error occurred while generating charts: {e}")  # Error handling during chart creation

# Function to generate downloadable data as CSV
def download_data(df, file_name="Updated_Data.csv"):
    return df.to_csv(index=False).encode('utf-8')  # Convert dataframe to CSV format and encode it

# --- Data Cleaning and Missing Value Handling ---
# Check for missing values in the dataframe and summarize them
missing_summary = df.isnull().sum().reset_index()  # Get the count of missing values per column
missing_summary.columns = ["Column", "Missing Values"]  # Rename columns for better clarity
missing_summary["% Missing"] = (missing_summary["Missing Values"] / len(df)) * 100  # Calculate the percentage of missing values per column

st.header("2. Data Cleaning")

# Check if there are any missing values
if missing_summary["Missing Values"].sum() > 0:
    st.write("### Missing Values Detected")  # Display message if missing values exist
    # Show missing value summary with a red background gradient to highlight the missing values
    st.write(missing_summary.style.background_gradient(cmap="Reds"))
    # Display rows with missing values
    missing_rows = df[df.isnull().any(axis=1)]
    st.write("Rows with Missing Values")
    st.dataframe(missing_rows)
    
    # Prompt the user with options to handle missing values
    missing_action = st.radio(
        "How would you like to handle missing values?",  # Radio button for missing value handling options
        (
            "Drop Rows with Missing Values",  # Option 1: Drop rows containing missing values
            "Fill Missing Values with Mean (Numeric Columns)",  # Option 2: Fill missing numeric values with mean
            "Fill Missing Values with Median (Numeric Columns)",  # Option 3: Fill missing numeric values with median
            "Fill Missing Values with Mode (Categorical Columns)",  # Option 4: Fill missing categorical values with mode
            "Fill Missing Values with Custom Value",  # Option 5: Fill missing values with a user-defined custom value
        ),
    )
    
    # Handling missing data based on the user's selection
    if missing_action == "Drop Rows with Missing Values":
        df = df.dropna()  # Drop rows with any missing values
        st.write("Rows with missing values dropped.")  # Inform the user that missing rows have been dropped
    elif missing_action == "Fill Missing Values with Mean (Numeric Columns)":
        # Fill missing values in numeric columns with the mean of that column
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns  # Select numeric columns
        for col in numeric_columns:
            df[col].fillna(df[col].mean(), inplace=True)  # Fill missing numeric values with the mean
        st.write("Missing values in numeric columns filled with the mean.")  # Inform the user about the action taken
    elif missing_action == "Fill Missing Values with Median (Numeric Columns)":
        # Fill missing values in numeric columns with the median of that column
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns  # Select numeric columns
        for col in numeric_columns:
            df[col].fillna(df[col].median(), inplace=True)  # Fill missing numeric values with the median
        st.write("Missing values in numeric columns filled with the median.")  # Inform the user about the action taken
    elif missing_action == "Fill Missing Values with Mode (Categorical Columns)":
        # Fill missing values in categorical columns with the mode (most frequent value) of that column
        categorical_columns = df.select_dtypes(include=["object"]).columns  # Select categorical columns
        for col in categorical_columns:
            df[col].fillna(df[col].mode()[0], inplace=True)  # Fill missing categorical values with the mode
        st.write("Missing values in categorical columns filled with the mode.")  # Inform the user about the action taken
    elif missing_action == "Fill Missing Values with Custom Value":
        # Prompt the user for a custom value to fill missing values
        custom_value = st.text_input("Enter a custom value to fill missing values:", "Unknown")  # Default custom value is 'Unknown'
        # Fill all columns with the custom value
        df.fillna(custom_value, inplace=True)
        st.write(f"Missing values filled with '{custom_value}'.")  # Inform the user that missing values are filled with the custom value
else:
    st.write("### No Missing Values Detected")  # Display message if no missing values are found

# Handle 'NaN' values in 'Category' and 'Sub-Category' for visualization purposes (treemap)
df['Category'] = df['Category'].fillna('Unknown')  # Fill missing 'Category' values with 'Unknown'
df['Sub-Category'] = df['Sub-Category'].fillna('Unknown')  # Fill missing 'Sub-Category' values with 'Unknown'

# Display the cleaned data to the user
st.subheader("Raw Data")
data_display_option = st.radio(
    "Choose how to view the raw data:",  # Allow the user to choose how to view the raw data
    options=["Display All Rows", "Display First 100 Rows", "Interactive Table"],  # Options to display the data
    horizontal=True  # Display options horizontally
)

# Display the raw data based on the user's selection
if data_display_option == "Display All Rows":
    st.dataframe(df)  # Show all rows of the cleaned data
elif data_display_option == "Display First 100 Rows":
    st.dataframe(df.head(100))  # Show the first 100 rows of the cleaned data
else:
    AgGrid(df)  # Display an interactive table for the raw data (if AgGrid is installed)


# Data Filtering Section
st.header("3. Data Filtering")
# Convert the 'Order Date' column to datetime format to enable time-based filtering
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
# Date range filtering options
col1, col2 = st.columns(2)  # Create two columns for date input widgets
startDate = df["Order Date"].min()  # Default start date as the minimum order date in the data
endDate = df["Order Date"].max()  # Default end date as the maximum order date in the data
# Date input widgets for filtering the data based on user input
with col1:
    date1 = st.date_input("Start Date", startDate)  # Start date input
with col2:
    date2 = st.date_input("End Date", endDate)  # End date input
# Filter the data based on the selected date range
df = df[(df["Order Date"] >= pd.to_datetime(date1)) & (df["Order Date"] <= pd.to_datetime(date2))]

# Sidebar filters for selecting region, state, and city
st.sidebar.header("Choose your filter:")
region = st.sidebar.multiselect("Pick your Region", options=df["Region"].unique())  # Region filter
state = st.sidebar.multiselect("Pick the State", options=df[df["Region"].isin(region) if region else slice(None)]["State"].unique())  # State filter based on selected region
city = st.sidebar.multiselect("Pick the City", options=df[df["State"].isin(state) if state else slice(None)]["City"].unique())  # City filter based on selected state
filtered_df = filter_by_location(df, region, state, city)  # Apply the location filter to the dataframe

# --- Visualizations with View and Download Options ---
st.markdown("<h1 style='font-size:50px;'>DATA VISUALIZATION</h1>", unsafe_allow_html=True)
st.header("4. Visualizations")

# Category-wise Sales Visualization
st.subheader("Category-wise Sales")
fig = plot_category_sales(filtered_df)  # Plot category-wise sales
st.plotly_chart(fig, use_container_width=True, key="category_sales_chart")  # Display chart
# View Data Option for category-wise sales
with st.expander("View Category-wise Data"):
    st.dataframe(filtered_df[['Category', 'Sales']])  # Show a preview of the data
# Download option for category-wise sales data
category_csv = filtered_df[['Category', 'Sales']].to_csv(index=False)
st.download_button(
    label="Download Category-wise Sales Data",
    data=category_csv,
    file_name="Category_Sales.csv",
    mime="text/csv"
)

# Region-wise Sales Visualization
st.subheader("Region-wise Sales")
fig = plot_region_sales(filtered_df)  # Plot region-wise sales
st.plotly_chart(fig, use_container_width=True, key="region_sales_chart")  # Display chart
# View Data Option for region-wise sales
with st.expander("View Region-wise Data"):
    st.dataframe(filtered_df[['Region', 'Sales']])  # Show a preview of the data
# Download option for region-wise sales data
region_csv = filtered_df[['Region', 'Sales']].to_csv(index=False)
st.download_button(
    label="Download Region-wise Sales Data",
    data=region_csv,
    file_name="Region_Sales.csv",
    mime="text/csv"
)

# Time Series Analysis Visualization
st.subheader("Time Series Analysis")
fig = plot_time_series_sales(filtered_df)  # Plot time series sales
st.plotly_chart(fig, use_container_width=True, key="time_series_chart")  # Display chart
# View Data Option for time series sales
with st.expander("View Time Series Data"):
    st.dataframe(filtered_df[['month_year', 'Sales']])  # Show a preview of the data
# Download option for time series sales data
time_series_csv = filtered_df[['month_year', 'Sales']].to_csv(index=False)
st.download_button(
    label="Download Time Series Sales Data",
    data=time_series_csv,
    file_name="Time_Series_Sales.csv",
    mime="text/csv"
)

# Treemap Visualization for hierarchical sales view
st.subheader("Hierarchical View of Sales")
fig = plot_sales_treemap(filtered_df)  # Plot sales treemap
st.plotly_chart(fig, use_container_width=True, key="treemap_sales_chart")  # Display chart
# View Data Option for treemap sales
with st.expander("View Treemap Data"):
    st.dataframe(filtered_df[['Region', 'Category', 'Sub-Category', 'Sales']])  # Show a preview of the data
# Download option for treemap data
treemap_csv = filtered_df[['Region', 'Category', 'Sub-Category', 'Sales']].to_csv(index=False)
st.download_button(
    label="Download Treemap Sales Data",
    data=treemap_csv,
    file_name="Treemap_Sales.csv",
    mime="text/csv"
)

# Scatter Plot for relationship between sales and profit
st.subheader("Relationship Between Sales and Profit")
fig = plot_scatter_sales_profit(filtered_df)  # Plot scatter plot
st.plotly_chart(fig, use_container_width=True, key="scatter_sales_profit_chart")  # Display chart
# View Data Option for sales and profit
with st.expander("View Sales and Profit Data"):
    st.dataframe(filtered_df[['Sales', 'Profit']])  # Show a preview of the data
# Download option for scatter plot data
scatter_csv = filtered_df[['Sales', 'Profit']].to_csv(index=False)
st.download_button(
    label="Download Sales and Profit Data",
    data=scatter_csv,
    file_name="Sales_Profit_Data.csv",
    mime="text/csv"
)

# Viewing first 500 rows of filtered data
st.subheader("First 500 Rows of Filtered Data")
first_500_rows = filtered_df.head(500)  # Extract first 500 rows for preview
st.dataframe(first_500_rows)  # Display first 500 rows
# Download button for the first 500 rows
st.download_button(
    label="Download First 500 Rows",
    data=first_500_rows.to_csv(index=False),
    file_name="First_500_Rows.csv",
    mime="text/csv"
)

# --- Month-wise Sub-Category Sales Summary ---
st.subheader(":point_right: Month-wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region", "State", "City", "Category", "Sales", "Profit", "Quantity"]]  # Sample data for table
    fig = ff.create_table(df_sample, colorscale="Cividis")  # Create table visualization
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Month-wise Sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()  # Extract month names from order date
    sub_category_Year = pd.pivot_table(data=filtered_df, values="Sales", index=["Sub-Category"], columns="month")  # Pivot table for sales by sub-category and month
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))  # Display the pivot table with color gradient

# Data Preprocessing Section for missing values handling
st.markdown("<h1 style='font-size:50px;'>SALES FORECASTING</h1>", unsafe_allow_html=True)
with st.expander("Data Preprocessing"):
    if df is not None:
        missing_values_option = st.selectbox("How to handle missing values?", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])  # Select missing value handling method
        if missing_values_option == "Drop rows":
            df.dropna(inplace=True)  # Drop rows with missing values
        elif missing_values_option == "Fill with mean":
            df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean
        elif missing_values_option == "Fill with median":
            df.fillna(df.median(), inplace=True)  # Fill missing values with column median
        elif missing_values_option == "Fill with mode":
            df.fillna(df.mode().iloc[0], inplace=True)  # Fill missing values with column mode
        st.write("Missing values handled.")  # Confirmation message
        st.dataframe(df.head(100))  # Display first 100 rows of the cleaned data

# Visualization Filters for user interaction
st.subheader("Filters for Visualization")
col1, col2 = st.columns(2)  # Create two columns for filters
with col1:
    region_filter = st.selectbox("Select Region", options=df['Region'].unique())  # Region filter
with col2:
    category_filter = st.selectbox("Select Category", options=df['Category'].unique())  # Category filter
# Filter the data based on user input
filtered_data = df[(df['Region'] == region_filter) & (df['Category'] == category_filter)]  # Apply the region and category filters
st.write(f"Displaying data for Region: {region_filter} and Category: {category_filter}")  # Display filter info
st.dataframe(filtered_data.head(100))  # Display the filtered data

# Data Visualizations for filtered data
st.subheader("Visualizations")
if df is not None:
    create_charts(filtered_data)  # Call the function to create visualizations for the filtered data

# Forecasting Section for future sales prediction
st.subheader("Future Sales Prediction")
if df is not None:
    forecast_period = st.slider("Select forecast period (months)", min_value=1, max_value=24, value=12)  # Select forecast period
    forecast_sales(filtered_data, forecast_period)  # Predict future sales based on selected forecast period

# Download Section for processed and filtered data
st.download_button("Download Processed Data", df.to_csv(index=False), file_name="processed_superstore.csv", mime="text/csv")  # Download processed data
st.download_button("Download Filtered Data", filtered_data.to_csv(index=False), file_name="filtered_superstore.csv", mime="text/csv")  # Download filtered data
