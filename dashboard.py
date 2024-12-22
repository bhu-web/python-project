# importing necessary libraries
import streamlit as st     #for creating interactive dashboard
import plotly.express as px      #for creating plots(graphs)
import pandas as pd     #for data manipulation
import os     #for file operations
import warnings     #to supress warnings
from plotly import graph_objects as go     #for more advanced plotting
warnings.filterwarnings('ignore')     #ignore any warnings during execution

# Setting the page configuration for the Streamlit app
st.set_page_config(page_title="Superstore EDA", page_icon=":bar_chart:", layout="wide")

# Page header with title
st.title(":bar_chart: Sample Superstore EDA")

# File uploader widget to upload a dataset file
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "txt", "xlsx", "xls"])

# If a file is uploaded, read the file into a DataFrame
if fl is not None:
    try:
        # Try reading the uploaded file
        df = pd.read_excel(fl)
    except Exception as e:
        # If there's an error in reading the file, show an error message
        st.error(f"Error reading the file: {e}")
        st.stop()
else:
    # If no file is uploaded, use a default file path
    default_path = r"C:\Users\bhoom\OneDrive\Documents\opencv\myenv\Superstore.csv"
    if os.path.exists(default_path):
        df = pd.read_excel(default_path, encoding="ISO-8859-1")     # Reading the default file
    else:
        st.error("Default file not found. Please upload a file.")
        st.stop()

# Convert the 'Order Date' column to datetime format
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Date range filtering options
col1, col2 = st.columns((2))     # Create two columns for date inputs
startDate = df["Order Date"].min()     # Set the default start date as the min order date
endDate = df["Order Date"].max()     # Set the default end date as the max order date

# Date input widgets for filtering
with col1:
    date1 = st.date_input("Start Date", startDate)
with col2:
    date2 = st.date_input("End Date", endDate)

# Filter data based on selected date range
df = df[(df["Order Date"] >= pd.to_datetime(date1)) & (df["Order Date"] <= pd.to_datetime(date2))]

# Sidebar filters for Region, State, and City
st.sidebar.header("Choose your filter:")
region = st.sidebar.multiselect("Pick your Region", options=df["Region"].unique())
state = st.sidebar.multiselect("Pick the State", options=df[df["Region"].isin(region) if region else slice(None)]["State"].unique())
city = st.sidebar.multiselect("Pick the City", options=df[df["State"].isin(state) if state else slice(None)]["City"].unique())

# Filter data based on the selected region, state, and city
if not region and not state and not city:
    filtered_df = df
elif not state and not city:
    filtered_df = df[df["Region"].isin(region)]
elif not region and not city:
    filtered_df = df[df["State"].isin(state)]
elif state and city:
    filtered_df = df3[df3["State"].isin(state) & df3["City"].isin(city)]
elif region and city:
    filtered_df = df3[df3["Region"].isin(region) & df3["City"].isin(city)]
elif region and state:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state)]
elif city:
    filtered_df = df3[df3["City"].isin(city)]
else:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]


# Category-wise Sales visualization 
category_df = filtered_df.groupby("Category", as_index=False)["Sales"].sum()
fig = px.bar(category_df, x="Category", y="Sales", text=category_df["Sales"].map("${:,.2f}".format))
st.subheader("Category-wise Sales")
st.plotly_chart(fig, use_container_width=True)

# Region-wise Sales visualization (pie chart)
fig = px.pie(filtered_df, values="Sales", names="Region", hole=0.5)
st.subheader("Region-wise Sales")
st.plotly_chart(fig, use_container_width=True)

# Time series analysis for sales over time
filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")     # Extract month and year
linechart = (
    filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y-%b"))["Sales"]
    .sum()
    .reset_index()
)
fig = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"})
st.subheader("Time Series Analysis")
st.plotly_chart(fig, use_container_width=True)

# Option to view and download the time series data
with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))     # Display data with gradient
    csv = linechart.to_csv(index=False).encode("utf-8")     # Convert data to CSV format
    st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')     # Download button for CSV

# Hierarchical view of sales using Treemap
st.subheader("Hierarchical View of Sales")
fig = px.treemap(
    filtered_df,
    path=["Region", "Category", "Sub-Category"],     # Hierarchical levels
    values="Sales",     # Values to size the blocks
    color="Sub-Category",     # Color by sub-category
)
st.plotly_chart(fig, use_container_width=True)

# Segment-wise and Category-wise Sales (Pie charts)
chart1, chart2 = st.columns((2))     # Create two columns for side-by-side charts
with chart1:
    st.subheader('Segment wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Segment", template = "plotly_dark")     # Pie chart for Segment
    fig.update_traces(text = filtered_df["Segment"], textposition = "inside")     # Display segment names inside
    st.plotly_chart(fig,use_container_width=True)

with chart2:
    st.subheader('Category wise Sales')
    fig = px.pie(filtered_df, values = "Sales", names = "Category", template = "gridon")     # Pie chart for Category
    fig.update_traces(text = filtered_df["Category"], textposition = "inside")     # Display category names inside
    st.plotly_chart(fig,use_container_width=True)

# Scatter plot showing the relationship between Sales and Profit
st.subheader("Relationship Between Sales and Profit")
fig = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity", color="Category")     # Scatter plot
st.plotly_chart(fig, use_container_width=True)

# Option to view the first 500 rows of data with a background gradient
with st.expander("View Data"):
    st.write(filtered_df.iloc[:500,1:20:2].style.background_gradient(cmap="Oranges"))     # Display sample data

# Month-wise Sub-Category Sales Summary
import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-Category Sales Summary")
with st.expander("Summary_Table"):
    df_sample = df[0:5][["Region","State","City","Category","Sales","Profit","Quantity"]]     # Sample data
    fig = ff.create_table(df_sample, colorscale = "Cividis")     # Create a table for sample data
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month wise sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()     # Extract month name
    sub_category_Year = pd.pivot_table(data = filtered_df, values = "Sales", index = ["Sub-Category"],columns = "month")     # Pivot table
    st.write(sub_category_Year.style.background_gradient(cmap="Blues"))     # Display with gradient

# Option to download the original dataset as CSV
csv = df.to_csv(index = False).encode('utf-8')     # Convert the entire dataset to CSV
st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")     # Download button for CSV
