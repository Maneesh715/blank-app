import streamlit as st
import pandas as pd

# Load data from Google Sheets (Replace with your CSV link)
excel_url = "https://docs.google.com/spreadsheets/d/1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo/edit?output=xlsx"
df = pd.read_excel(excel_url)

# Convert 'Month-Year' to datetime for sorting
df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
selected_months = st.sidebar.multiselect("Select Month-Year", sorted(df['Month-Year'].dt.strftime('%b-%Y').unique()), default=None)
selected_deal_managers = st.sidebar.multiselect("Select Deal Manager", df['Deal Manager'].unique())
selected_customers = st.sidebar.multiselect("Select Customer", df['Customer'].unique())
selected_countries = st.sidebar.multiselect("Select Country", df['Country'].unique())
selected_plants = st.sidebar.multiselect("Select Plant Type", df['Plant Type'].unique())

# Apply filters
filtered_df = df.copy()

if selected_months:
    filtered_df = filtered_df[filtered_df['Month-Year'].dt.strftime('%b-%Y').isin(selected_months)]
if selected_deal_managers:
    filtered_df = filtered_df[filtered_df['Deal Manager'].isin(selected_deal_managers)]
if selected_customers:
    filtered_df = filtered_df[filtered_df['Customer'].isin(selected_customers)]
if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
if selected_plants:
    filtered_df = filtered_df[filtered_df['Plant Type'].isin(selected_plants)]

st.title("📊 Sales Performance Dashboard")

# Grouping options
group_by = st.selectbox("Group By", ['Month-Year', 'Deal Manager', 'Customer', 'Country', 'Plant Type'])

# Aggregate Metrics
agg_metrics = {
    'Committed Orders': 'sum',
    'Achieved Orders': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Gross Margin': 'sum',
    'Achieved Gross Margin': 'sum'
}

# Perform aggregation
grouped = filtered_df.groupby(group_by).agg(agg_metrics).reset_index()

# Sort by Month-Year if selected
if group_by == 'Month-Year':
    grouped = grouped.sort_values(by='Month-Year')

# Display table
st.dataframe(grouped)

# Optional: Plot Charts
st.subheader(f"📈 {group_by}-wise Revenue Comparison")
st.bar_chart(grouped.set_index(group_by)[['Committed Revenue', 'Achieved Revenue']])

st.subheader(f"📈 {group_by}-wise Orders Comparison")
st.bar_chart(grouped.set_index(group_by)[['Committed Orders', 'Achieved Orders']])

