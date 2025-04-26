import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# Set Streamlit page config
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("🚀 Sales Dashboard")

# Safe division function
def safe_divide(numerator, denominator):
    return (numerator / denominator * 100) if denominator != 0 else 0

# Load and process data
@st.cache_data(ttl=3600)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo/export?format=xlsx"
    xls = pd.read_excel(url, sheet_name=None, engine="openpyxl")
    sheet1 = xls['Sheet1']
    sheet2 = xls['Sheet2']
    sheet3 = xls['Sheet3']

    df = pd.merge(sheet1, sheet2, on=['Month-Year', 'Deal Manager', 'Deal ID', 'Customer', 'New Customer', 'Country', 'Plant Type'], how='outer')

    sheet3['Committed Gross Margin'] = sheet3['Committed Revenue'] - (
        sheet3['Committed COGS'] + sheet3['Committed Logistics'] + sheet3['Committed P&F']
    )
    sheet3['Achieved Gross Margin'] = sheet3['Achieved Revenue'] - (
        sheet3['Achieved COGS'] + sheet3['Achieved Logistics'] + sheet3['Achieved P&F']
    )

    df = pd.merge(df, sheet3[['Month-Year', 'Deal Manager', 'Deal ID', 'Customer', 'New Customer', 'Country', 'Plant Type', 
                              'Committed Gross Margin', 'Achieved Gross Margin']],
                  on=['Month-Year', 'Deal Manager', 'Deal ID', 'Customer', 'New Customer', 'Country', 'Plant Type'], how='outer')

    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')
    df['Financial Year'] = df['Month-Year'].apply(lambda x: f"FY{x.year+1}" if x.month >= 4 else f"FY{x.year}")
    
    # Calculate Gross Margin % fields
    df['Committed GM %'] = df.apply(lambda row: safe_divide(row['Committed Gross Margin'], row['Committed Revenue']), axis=1)
    df['Achieved GM %'] = df.apply(lambda row: safe_divide(row['Achieved Gross Margin'], row['Achieved Revenue']), axis=1)

    return df

df = load_data()

# Sidebar Filters
st.sidebar.title("🔍 Filters")
filter_cols = ['Financial Year', 'Deal Manager', 'Customer', 'New Customer', 'Country', 'Plant Type']

filters = {}
for col in filter_cols:
    options = df[col].dropna().unique().tolist()
    selection = st.sidebar.multiselect(f"Select {col}", options, default=options)
    filters[col] = selection

# Apply filters
filtered_df = df.copy()
for col, selected_vals in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# Chart Mode Toggle (Grouped / Stacked)
chart_mode = st.radio("Chart Mode", ['Grouped', 'Stacked'], horizontal=True)

# Gross Margin View Toggle (Value / %)
gm_view = st.radio("Gross Margin View", ['By Value', 'By %'], horizontal=True)

# Helper to draw bar charts
def draw_bar_chart(df, y1, y2, title, yaxis_title):
    df_grouped = df.groupby('Financial Year')[[y1, y2]].sum().reset_index()
    df_melted = df_grouped.melt(id_vars='Financial Year', value_vars=[y1, y2], var_name='Metric', value_name='Value')
    barmode = 'group' if chart_mode == 'Grouped' else 'stack'
    fig = px.bar(df_melted, x='Financial Year', y='Value', color='Metric', barmode=barmode, title=title)
    fig.update_layout(yaxis_title=yaxis_title)
    return fig

# Calculate Summary for KPIs
summary = filtered_df.groupby('Financial Year').agg({
    'Achieved Orders': 'sum',
    'Committed Orders': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Gross Margin': 'sum',
    'Committed Gross Margin': 'sum',
    'Achieved GM %': 'mean',
    'Committed GM %': 'mean'
}).reset_index()

# Top KPI cards
st.subheader("📈 Overall Performance KPIs")

total_committed_orders = summary['Committed Orders'].sum()
total_achieved_orders = summary['Achieved Orders'].sum()

total_committed_revenue = summary['Committed Revenue'].sum()
total_achieved_revenue = summary['Achieved Revenue'].sum()

total_committed_gm = summary['Committed Gross Margin'].sum()
total_achieved_gm = summary['Achieved Gross Margin'].sum()

avg_committed_gm_pct = summary['Committed GM %'].mean()
avg_achieved_gm_pct = summary['Achieved GM %'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Orders (Achieved vs Committed)", f"{total_achieved_orders:,.0f}", f"{safe_divide(total_achieved_orders, total_committed_orders) - 100:.2f}%")
col2.metric("Revenue (Achieved vs Committed)", f"${total_achieved_revenue:,.0f}", f"{safe_divide(total_achieved_revenue, total_committed_revenue) - 100:.2f}%")
col3.metric("Gross Margin Value", f"${total_achieved_gm:,.0f}", f"{safe_divide(total_achieved_gm, total_committed_gm) - 100:.2f}%")
col4.metric("Gross Margin %", f"{avg_achieved_gm_pct:.2f}%", f"{avg_achieved_gm_pct - avg_committed_gm_pct:.2f}%")

# Orders Comparison
st.subheader("📦 Orders Comparison")
st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Orders', 'Achieved Orders', "Orders Comparison", "Orders"))

# Revenue Comparison
st.subheader("💰 Revenue Comparison")
st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Revenue', 'Achieved Revenue', "Revenue Comparison", "Revenue"))

# Gross Margin Comparison
st.subheader("📈 Gross Margin Comparison")
if gm_view == 'By Value':
    st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Gross Margin', 'Achieved Gross Margin', "Gross Margin (Value)", "Gross Margin Value"))
else:
    df_gm = filtered_df.groupby('Financial Year')[['Committed GM %', 'Achieved GM %']].mean().reset_index()
    df_melted = df_gm.melt(id_vars='Financial Year', value_vars=['Committed GM %', 'Achieved GM %'], var_name='Metric', value_name='Value')
    barmode = 'group' if chart_mode == 'Grouped' else 'stack'
    fig = px.bar(df_melted, x='Financial Year', y='Value', color='Metric', barmode=barmode, title="Gross Margin (%)")
    fig.update_layout(yaxis_title="Gross Margin %")
    st.plotly_chart(fig)

# Delta Summary Table
st.subheader("📊 Delta Summary Table (Achieved – Committed)")

summary['Delta Orders'] = summary['Achieved Orders'] - summary['Committed Orders']
summary['Delta Revenue'] = summary['Achieved Revenue'] - summary['Committed Revenue']
summary['Delta GM Value'] = summary['Achieved Gross Margin'] - summary['Committed Gross Margin']
summary['Delta GM %'] = summary['Achieved GM %'] - summary['Committed GM %']

delta_table = summary[['Financial Year', 'Delta Orders', 'Delta Revenue', 'Delta GM Value', 'Delta GM %']]
st.dataframe(delta_table.style.format({
    'Delta Orders': '{:,.0f}',
    'Delta Revenue': '{:,.0f}',
    'Delta GM Value': '{:,.0f}',
    'Delta GM %': '{:.2f}%'
}))

# Download Button
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Delta Summary')
    processed_data = output.getvalue()
    return processed_data

excel_data = convert_df_to_excel(delta_table)

st.download_button(
    label="📥 Download Delta Summary as Excel",
    data=excel_data,
    file_name="delta_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
