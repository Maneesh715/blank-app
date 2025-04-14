# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# 1. Set page title and layout
st.set_page_config(page_title="⚙️ IndustroDash", layout="wide", page_icon="📊")

# 2. Load Logo and Add Header
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("https://your-company-logo-url.com/logo.png", width=60)  # Replace with real logo URL
with col2:
    st.title("Worldref Sales Dashboard")

st.markdown("---")

@st.cache_data(ttl=3600)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo/export?format=xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')
    return df

df = load_data()

def safe_divide(numerator, denominator):
    return (numerator / denominator) * 100 if denominator else 0

# 3. Sidebar with Filters
st.sidebar.header("🔍 Filter Data")
selected_months = st.sidebar.multiselect("📅 Select Month-Year", sorted(df['Month-Year'].dt.strftime('%b-%Y').unique()))
selected_deal_managers = st.sidebar.multiselect("👨‍💼 Deal Manager", df['Deal Manager'].unique())
selected_customers = st.sidebar.multiselect("🏢 Customer", df['Customer'].unique())
selected_countries = st.sidebar.multiselect("🌍 Country", df['Country'].unique())
selected_plants = st.sidebar.multiselect("🏣 Plant Type", df['Plant Type'].unique())

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

# 4. Summary KPIs
st.subheader("📊 Performance Summary")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("💰 Total Achieved Revenue", f"${filtered_df['Achieved Revenue'].sum():,.0f}")
kpi2.metric("📦 Total Achieved Orders", f"{filtered_df['Achieved Orders'].sum():,.0f}")
kpi3.metric("📈 Achieved Gross Margin", f"${filtered_df['Achieved Gross Margin'].sum():,.0f}")

# 5. Grouped Aggregation
group_by = st.selectbox("📁 Group Data By", ['Month-Year', 'Deal Manager', 'Customer', 'Country', 'Plant Type'])

agg = {
    'Committed Orders': 'sum',
    'Achieved Orders': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Gross Margin': 'sum',
    'Achieved Gross Margin': 'sum'
}
grouped = filtered_df.groupby(group_by).agg(agg).reset_index()

if group_by == 'Month-Year':
    grouped[group_by] = pd.to_datetime(grouped[group_by], errors='coerce')
    grouped = grouped.sort_values(by=group_by)
    grouped[group_by] = grouped[group_by].dt.strftime('%b-%Y')

grouped['Revenue Conversion %'] = grouped.apply(lambda x: safe_divide(x['Achieved Revenue'], x['Committed Revenue']), axis=1)
grouped['Orders Conversion %'] = grouped.apply(lambda x: safe_divide(x['Achieved Orders'], x['Committed Orders']), axis=1)
grouped['GM Conversion %'] = grouped.apply(lambda x: safe_divide(x['Achieved Gross Margin'], x['Committed Gross Margin']), axis=1)

# 6. Table Display
st.subheader(f"📟 Detailed Performance by {group_by}")
styled = grouped.style.format({
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
)
st.dataframe(styled, use_container_width=True)

# 7. Download Excel Option
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Dashboard')
    return output.getvalue()

excel_data = convert_df_to_excel(grouped)
st.download_button(
    label="⬇️ Download Table as Excel",
    data=excel_data,
    file_name="industro_sales_dashboard.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# 8. Custom Plot Function with Branded Colors
def plot_chart(title, x, y, df, colors, ylabel):
    fig = px.bar(df, x=x, y=y, barmode="group", text_auto='.2s', color_discrete_sequence=colors)
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=ylabel,
        xaxis_tickangle=45,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        title_font=dict(size=18, color='white'),
        legend=dict(font=dict(size=12, color='white')),
        xaxis=dict(color='white', gridcolor='gray'),
        yaxis=dict(color='white', gridcolor='gray')
    )
    st.plotly_chart(fig, use_container_width=True)

# 9. Branded Graph Colors
st.subheader(f"💰 Revenue Comparison by {group_by}")
plot_chart("Revenue Comparison", group_by, ['Committed Revenue', 'Achieved Revenue'], grouped, ['#005B96', '#FFC20E'], "Revenue (USD)")

st.subheader(f"📦 Orders Comparison by {group_by}")
plot_chart("Orders Comparison", group_by, ['Committed Orders', 'Achieved Orders'], grouped, ['#5C4B99', '#00B159'], "Orders (Count)")

st.subheader(f"📈 Gross Margin Comparison by {group_by}")
plot_chart("Gross Margin Comparison", group_by, ['Committed Gross Margin', 'Achieved Gross Margin'], grouped, ['#D72638', '#17becf'], "Gross Margin (USD)")
