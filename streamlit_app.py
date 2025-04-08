import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
excel_url = "https://docs.google.com/spreadsheets/d/1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo/export?format=xlsx"
df = pd.read_excel(excel_url, engine='openpyxl')

# Convert Month-Year to datetime
df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
selected_months = st.sidebar.multiselect("📅 Select Month-Year", sorted(df['Month-Year'].dt.strftime('%b-%Y').unique()))
selected_deal_managers = st.sidebar.multiselect("👨‍💼 Select Deal Manager", df['Deal Manager'].unique())
selected_customers = st.sidebar.multiselect("🏢 Select Customer", df['Customer'].unique())
selected_countries = st.sidebar.multiselect("🌍 Select Country", df['Country'].unique())
selected_plants = st.sidebar.multiselect("🏭 Select Plant Type", df['Plant Type'].unique())

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

# Grouping selection
group_by = st.selectbox("📁 Group Data By", ['Month-Year', 'Deal Manager', 'Customer', 'Country', 'Plant Type'])

# Aggregate data
agg_metrics = {
    'Committed Orders': 'sum',
    'Achieved Orders': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Gross Margin': 'sum',
    'Achieved Gross Margin': 'sum'
}
grouped = filtered_df.groupby(group_by).agg(agg_metrics).reset_index()

# Sort Month-Year if selected
if group_by == 'Month-Year':
    grouped[group_by] = pd.to_datetime(grouped[group_by], format='%b-%Y')
    grouped = grouped.sort_values(by=group_by)
    grouped[group_by] = grouped[group_by].dt.strftime('%b-%Y')

# Conversion Rate Calculations
grouped['Revenue Conversion %'] = (grouped['Achieved Revenue'] / grouped['Committed Revenue']) * 100
grouped['Orders Conversion %'] = (grouped['Achieved Orders'] / grouped['Committed Orders']) * 100
grouped['GM Conversion %'] = (grouped['Achieved Gross Margin'] / grouped['Committed Gross Margin']) * 100

# Show data
st.dataframe(grouped.style.format({
    'Committed Revenue': '{:,.0f}',
    'Achieved Revenue': '{:,.0f}',
    'Revenue Conversion %': '{:.1f}%',
    'Committed Orders': '{:,.0f}',
    'Achieved Orders': '{:,.0f}',
    'Orders Conversion %': '{:.1f}%',
    'Committed Gross Margin': '{:,.0f}',
    'Achieved Gross Margin': '{:,.0f}',
    'GM Conversion %': '{:.1f}%'
}).applymap(
    lambda val: 'color: green' if isinstance(val, (int, float)) and val >= 100 else 'color: red',
    subset=['Revenue Conversion %', 'Orders Conversion %', 'GM Conversion %']
), use_container_width=True)

# ========== Revenue Comparison ==========
st.subheader(f"💰 Revenue Comparison by {group_by}")
fig_rev = px.bar(
    grouped,
    x=group_by,
    y=['Committed Revenue', 'Achieved Revenue'],
    barmode='group',
    text_auto='.2s',
    color_discrete_sequence=['#1f77b4', '#2ca02c'],
    height=500
)
fig_rev.update_layout(
    xaxis_title=group_by,
    yaxis_title="Revenue (USD)",
    xaxis_tickangle=45,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white',
    title_font=dict(size=16),
    legend=dict(font=dict(size=12))
)
st.plotly_chart(fig_rev, use_container_width=True)

# ========== Orders Comparison ==========
st.subheader(f"📦 Orders Comparison by {group_by}")
fig_orders = px.bar(
    grouped,
    x=group_by,
    y=['Committed Orders', 'Achieved Orders'],
    barmode='group',
    text_auto=True,
    color_discrete_sequence=['#ff7f0e', '#9467bd'],
    height=500
)
fig_orders.update_layout(
    xaxis_title=group_by,
    yaxis_title="Orders (Count)",
    xaxis_tickangle=45,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white',
    title_font=dict(size=16),
    legend=dict(font=dict(size=12))
)
st.plotly_chart(fig_orders, use_container_width=True)

# ========== Gross Margin Comparison ==========
st.subheader(f"📈 Gross Margin Comparison by {group_by}")
fig_gm = px.bar(
    grouped,
    x=group_by,
    y=['Committed Gross Margin', 'Achieved Gross Margin'],
    barmode='group',
    text_auto='.2s',
    color_discrete_sequence=['#d62728', '#17becf'],
    height=500
)
fig_gm.update_layout(
    xaxis_title=group_by,
    yaxis_title="Gross Margin (USD)",
    xaxis_tickangle=45,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='white',
    title_font=dict(size=16),
    legend=dict(font=dict(size=12))
)
st.plotly_chart(fig_gm, use_container_width=True)

