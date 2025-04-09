import streamlit as st
import pandas as pd
import plotly.express as px
import io

# === Authentication (Streamlit Community Cloud) ===
# Set up via https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/authentication

allowed_users = ["founder@example.com", "team@example.com"]
if st.session_state.get("authentication_status") != True:
    st.error("🔒 Please log in to access the dashboard.")
    st.stop()
elif st.session_state.get("username") not in allowed_users:
    st.error("⛔ Unauthorized access.")
    st.stop()

# === App Title ===
st.set_page_config(page_title="Sales Performance Dashboard", layout="wide")
st.title("📊 Sales Performance Dashboard")
st.markdown("Gain insights into order, revenue, and gross margin performance across time, customers, and teams.")

# === Load & Cache Data ===
@st.cache_data(ttl=3600)
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo/export?format=xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')
    return df

df = load_data()

# === Sidebar Filters ===
st.sidebar.header("🔍 Filter Data")
selected_months = st.sidebar.multiselect("📅 Select Month-Year", sorted(df['Month-Year'].dt.strftime('%b-%Y').unique()))
selected_deal_managers = st.sidebar.multiselect("👨‍💼 Deal Manager", df['Deal Manager'].unique())
selected_customers = st.sidebar.multiselect("🏢 Customer", df['Customer'].unique())
selected_countries = st.sidebar.multiselect("🌍 Country", df['Country'].unique())
selected_plants = st.sidebar.multiselect("🏭 Plant Type", df['Plant Type'].unique())

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

# === Grouping Logic ===
group_by = st.selectbox("📁 Group Data By", ['Month-Year', 'Deal Manager', 'Customer', 'Country', 'Plant Type'])

agg_metrics = {
    'Committed Orders': 'sum',
    'Achieved Orders': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Gross Margin': 'sum',
    'Achieved Gross Margin': 'sum'
}
grouped = filtered_df.groupby(group_by).agg(agg_metrics).reset_index()

# Sort month-year
if group_by == 'Month-Year':
    grouped[group_by] = pd.to_datetime(grouped[group_by], errors='coerce')
    grouped = grouped.sort_values(by=group_by)
    grouped[group_by] = grouped[group_by].dt.strftime('%b-%Y')

# === Conversion Metrics with NaN Protection ===
grouped['Revenue Conversion %'] = (grouped['Achieved Revenue'] / grouped['Committed Revenue'].replace(0, pd.NA)) * 100
grouped['Orders Conversion %'] = (grouped['Achieved Orders'] / grouped['Committed Orders'].replace(0, pd.NA)) * 100
grouped['GM Conversion %'] = (grouped['Achieved Gross Margin'] / grouped['Committed Gross Margin'].replace(0, pd.NA)) * 100

# === KPIs ===
col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Revenue", f"${grouped['Achieved Revenue'].sum():,.0f}")
col2.metric("📦 Total Orders", f"{grouped['Achieved Orders'].sum():,.0f}")
gm_pct = (grouped['Achieved Gross Margin'].sum() / grouped['Achieved Revenue'].sum()) * 100
col3.metric("📈 Gross Margin %", f"{gm_pct:.1f}%")

# === Styled Data Table ===
st.subheader("🧾 Summary Table")
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

# === Charting Function ===
def plot_chart(title, df, x, y, colors, yaxis_title):
    try:
        fig = px.bar(
            df, x=x, y=y, barmode='group', text_auto='.2s',
            color_discrete_sequence=colors, height=500
        )
        fig.update_layout(
            title=title,
            xaxis_title=x,
            yaxis_title=yaxis_title,
            xaxis_tickangle=45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(size=18),
            legend=dict(font=dict(size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render chart: {title}. Reason: {e}")

# === Charts ===
st.subheader(f"💰 Revenue Comparison by {group_by}")
plot_chart(
    title=f"Revenue Comparison by {group_by}",
    df=grouped,
    x=group_by,
    y=['Committed Revenue', 'Achieved Revenue'],
    colors=['#1f77b4', '#2ca02c'],
    yaxis_title="Revenue (USD)"
)

st.subheader(f"📦 Orders Comparison by {group_by}")
plot_chart(
    title=f"Orders Comparison by {group_by}",
    df=grouped,
    x=group_by,
    y=['Committed Orders', 'Achieved Orders'],
    colors=['#ff7f0e', '#9467bd'],
    yaxis_title="Orders (Count)"
)

st.subheader(f"📈 Gross Margin Comparison by {group_by}")
plot_chart(
    title=f"Gross Margin Comparison by {group_by}",
    df=grouped,
    x=group_by,
    y=['Committed Gross Margin', 'Achieved Gross Margin'],
    colors=['#d62728', '#17becf'],
    yaxis_title="Gross Margin (USD)"
)

# === Excel Download Button ===
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    grouped.to_excel(writer, index=False, sheet_name='Dashboard Data')

st.download_button(
    label="📥 Download Dashboard Data (Excel)",
    data=buffer.getvalue(),
    file_name="dashboard_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
