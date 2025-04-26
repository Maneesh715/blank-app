import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# Set Streamlit page config with a wider layout and title
st.set_page_config(page_title="Sales Dashboard", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f7f8fa;}
    .sidebar .sidebar-content {background-color: #4CAF50;}
    .stButton button {background-color: #4CAF50; color: white;}
    .stMetric .stMarkdown {font-size: 18px; color: #333;}
    .css-ffhzg2 {font-family: 'Arial', sans-serif;}
    .stBlockquote {background-color: #e7f3e1; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

# Title with Emoji and subtitle for a more attractive look
st.title("🚀 Sales Dashboard")
st.markdown("<h3 style='color: #4CAF50;'>Track your sales, margins, and revenue in real-time!</h3>", unsafe_allow_html=True)

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

    # Merging sheet1 with sheet2 using Deal ID as the unique key
    df = pd.merge(sheet1, sheet2[['Deal ID', 'Committed Revenue', 'Achieved Revenue']], on='Deal ID', how='outer')

    # Adding committed and achieved gross margin from sheet3
    sheet3['Committed Gross Margin'] = sheet2['Committed Revenue'] - (
        sheet3['Committed COGS'] + sheet3['Committed Logistics'] + sheet3['Committed P&F']
    )
    sheet3['Achieved Gross Margin'] = sheet2['Achieved Revenue'] - (
        sheet3['Achieved COGS'] + sheet3['Achieved Logistics'] + sheet3['Achieved P&F']
    )

    df = pd.merge(df, sheet3[['Deal ID', 'Committed Gross Margin', 'Achieved Gross Margin']],
                  on='Deal ID', how='outer')

    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')

    # Calculate Gross Margin % fields
    df['Committed GM %'] = df.apply(lambda row: safe_divide(row['Committed Gross Margin'], row['Committed Revenue']), axis=1)
    df['Achieved GM %'] = df.apply(lambda row: safe_divide(row['Achieved Gross Margin'], row['Achieved Revenue']), axis=1)

    return df

df = load_data()

# Sidebar Filters with custom styling and icons
st.sidebar.title("🔍 Filters")
filter_cols = ['Month-Year', 'Deal Manager', 'Customer', 'New Customer', 'Country', 'Plant Type']

filters = {}
for col in filter_cols:
    options = df[col].dropna().unique().tolist()
    selection = st.sidebar.multiselect(f"Select {col}", options, default=options)
    filters[col] = selection

# Apply filters, show overall data if no selection is made
filtered_df = df.copy()
for col, selected_vals in filters.items():
    if selected_vals:
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# Chart Mode Toggle (Grouped / Stacked)
chart_mode = st.radio("Chart Mode", ['Grouped', 'Stacked'], horizontal=True)

# Gross Margin View Toggle (Value / %)
gm_view = st.radio("Gross Margin View", ['By Value', 'By %'], horizontal=True)

# Helper to draw bar charts with custom colors and values on top
def draw_bar_chart(df, y1, y2, title, yaxis_title):
    df_grouped = df.groupby('Month-Year')[[y1, y2]].sum().reset_index()
    df_melted = df_grouped.melt(id_vars='Month-Year', value_vars=[y1, y2], var_name='Metric', value_name='Value')
    barmode = 'group' if chart_mode == 'Grouped' else 'stack'

    # Custom colors for Committed and Achieved
    custom_colors = {
        'Committed': '#007BFF',  # Blue for Committed
        'Achieved': '#28A745'    # Green for Achieved
    }

    fig = px.bar(df_melted, x='Month-Year', y='Value', color='Metric', barmode=barmode, title=title, 
                 color_discrete_map=custom_colors, text='Value')  # Show values on top of bars
    
    fig.update_layout(
        yaxis_title=yaxis_title, 
        plot_bgcolor="#F7F8FA", 
        template="plotly_dark",
        xaxis=dict(showgrid=True, zeroline=False),  # Stylish x-axis grid
        yaxis=dict(showgrid=True, zeroline=False),  # Stylish y-axis grid
        title_font=dict(size=18, family='Arial', color='white'),  # Title Font
        legend_title=dict(text="Metrics", font=dict(size=14, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        margin=dict(t=60, b=40, l=40, r=40)  # Adjust margins
    )
    
    # Enable values to appear on top of bars
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    return fig

# Calculate Summary for KPIs
summary = filtered_df.groupby('Month-Year').agg({
    'Achieved Orders': 'sum',
    'Committed Orders': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Gross Margin': 'sum',
    'Committed Gross Margin': 'sum',
    'Achieved GM %': 'mean',
    'Committed GM %': 'mean'
}).reset_index()

# Top KPI cards with enhanced styling
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

# Orders Comparison with chart
st.subheader("📦 Orders Comparison")
st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Orders', 'Achieved Orders', "Orders Comparison", "Orders"))

# Revenue Comparison with chart
st.subheader("💰 Revenue Comparison")
st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Revenue', 'Achieved Revenue', "Revenue Comparison", "Revenue"))

# Gross Margin Comparison with view toggle
st.subheader("📈 Gross Margin Comparison")
if gm_view == 'By Value':
    st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Gross Margin', 'Achieved Gross Margin', "Gross Margin (Value)", "Gross Margin Value"))
else:
    df_gm = filtered_df.groupby('Month-Year')[['Committed GM %', 'Achieved GM %']].mean().reset_index()
    df_melted = df_gm.melt(id_vars='Month-Year', value_vars=['Committed GM %', 'Achieved GM %'], var_name='Metric', value_name='Value')
    barmode = 'group' if chart_mode == 'Grouped' else 'stack'
    fig = px.bar(df_melted, x='Month-Year', y='Value', color='Metric', barmode=barmode, title="Gross Margin (%)",
                 color_discrete_map={'Committed GM %': '#007BFF', 'Achieved GM %': '#28A745'}, text='Value')
    fig.update_layout(yaxis_title="Gross Margin %")
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')  # Display values on top of the bars
    st.plotly_chart(fig)

# Delta Summary Table with improved styling
st.subheader("📊 Delta Summary Table (Achieved – Committed)")

summary['Delta Orders'] = summary['Achieved Orders'] - summary['Committed Orders']
summary['Delta Revenue'] = summary['Achieved Revenue'] - summary['Committed Revenue']
summary['Delta GM Value'] = summary['Achieved Gross Margin'] - summary['Committed Gross Margin']
summary['Delta GM %'] = summary['Achieved GM %'] - summary['Committed GM %']

delta_table = summary[['Month-Year', 'Delta Orders', 'Delta Revenue', 'Delta GM Value', 'Delta GM %']]
st.dataframe(delta_table.style.format({
    'Delta Orders': '{:,.0f}',
    'Delta Revenue': '{:,.0f}',
    'Delta GM Value': '{:,.0f}',
    'Delta GM %': '{:.2f}%'
}))

# Download Button with icon
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
