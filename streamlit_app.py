import streamlit as st
import pandas as pd
import plotly.express as px

# Load data from Google Sheets (Replace with your CSV link)
excel_url = "https://docs.google.com/spreadsheets/d/1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo/export?format=xlsx"
df = pd.read_excel(excel_url, engine='openpyxl')

# Convert 'Month-Year' to datetime for sorting
df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
selected_months = st.sidebar.multiselect("Select Month-Year", sorted(df['Month-Year'].dt.strftime('%b-%Y').unique()))
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

# Title
st.title("📊 Sales Performance Dashboard")

# Grouping options
group_by = st.selectbox("Group By", ['Month-Year', 'Deal Manager', 'Customer', 'Country', 'Plant Type'])

# Aggregation
agg_metrics = {
    'Committed Orders': 'sum',
    'Achieved Orders': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Gross Margin': 'sum',
    'Achieved Gross Margin': 'sum'
}
grouped = filtered_df.groupby(group_by).agg(agg_metrics).reset_index()

# Format Month-Year if needed
if group_by == 'Month-Year':
    grouped = grouped.sort_values(by='Month-Year')
    grouped['Month-Year'] = grouped['Month-Year'].dt.strftime('%b-%Y')
    grouped['Month-Year'] = pd.to_datetime(grouped['Month-Year'], format='%b-%Y')
    grouped = grouped.sort_values(by='Month-Year')
    grouped[group_by] = grouped['Month-Year'].dt.strftime('%b-%Y')

# Show table
st.dataframe(grouped)

# =================== Revenue Chart ===================
st.subheader(f"📈 {group_by}-wise Revenue Comparison")

if chart_type == 'Bar Chart':
    fig_rev = px.bar(
        grouped,
        x=group_by,
        y=['Committed Revenue', 'Achieved Revenue'],
        barmode='group',
        title='Revenue Comparison',
        color_discrete_sequence=['#1f77b4', '#2ca02c'],
        height=500,
        text_auto=True
    )
else:
    fig_rev = px.line(
        grouped,
        x=group_by,
        y=['Committed Revenue', 'Achieved Revenue'],
        title='Revenue Comparison',
        markers=True,
        color_discrete_sequence=['#1f77b4', '#2ca02c'],
        height=500
    )

fig_rev.update_layout(
    xaxis_title=group_by,
    yaxis_title="Revenue",
    xaxis_tickangle=60,
    xaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
    yaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
    title_font=dict(size=16),
    legend=dict(font=dict(size=12)),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white'
)
st.plotly_chart(fig_rev, use_container_width=True)

# =================== Orders Chart ===================
st.subheader(f"📈 {group_by}-wise Orders Comparison")

if chart_type == 'Bar Chart':
    fig_orders = px.bar(
        grouped,
        x=group_by,
        y=['Committed Orders', 'Achieved Orders'],
        barmode='group',
        title='Orders Comparison',
        color_discrete_sequence=['#ff7f0e', '#9467bd'],
        height=500,
        text_auto=True
    )
else:
    fig_orders = px.line(
        grouped,
        x=group_by,
        y=['Committed Orders', 'Achieved Orders'],
        title='Orders Comparison',
        markers=True,
        color_discrete_sequence=['#ff7f0e', '#9467bd'],
        height=500
    )

fig_orders.update_layout(
    xaxis_title=group_by,
    yaxis_title="Orders",
    xaxis_tickangle=60,
    xaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
    yaxis=dict(tickfont=dict(size=12), titlefont=dict(size=14)),
    title_font=dict(size=16),
    legend=dict(font=dict(size=12)),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white'
)
st.plotly_chart(fig_orders, use_container_width=True)
