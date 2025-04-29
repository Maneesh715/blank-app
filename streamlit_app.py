import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Orders Dashboard", layout="wide")
st.title("📊 Worldref Sales Dashboard")

# --- LOAD DATA FROM GOOGLE SHEET (as CSV export) ---
SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
SHEET_NAME = "Sheet1"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data(CSV_URL)

# --- CLEAN & PROCESS DATA ---
df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y")
df["New Customer"] = df["New Customer"].fillna(0).astype(int)
df["Committed Orders"] = pd.to_numeric(df["Committed Orders"], errors='coerce')
df["Achieved Orders"] = pd.to_numeric(df["Achieved Orders"], errors='coerce')
df["Conversion Rate (%)"] = (df["Achieved Orders"] / df["Committed Orders"]) * 100

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔎 Filters")
deal_managers = st.sidebar.multiselect("Select Deal Manager(s):", options=sorted(df["Deal Manager"].dropna().unique()), default=None)
countries = st.sidebar.multiselect("Select Country(ies):", options=sorted(df["Country"].dropna().unique()), default=None)
plants = st.sidebar.multiselect("Select Plant Type(s):", options=sorted(df["Plant Type"].dropna().unique()), default=None)
customers = st.sidebar.multiselect("Select Customer(s):", options=sorted(df["Customer"].dropna().unique()), default=None)

filtered_df = df.copy()
if deal_managers:
    filtered_df = filtered_df[filtered_df["Deal Manager"].isin(deal_managers)]
if countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
if plants:
    filtered_df = filtered_df[filtered_df["Plant Type"].isin(plants)]
if customers:
    filtered_df = filtered_df[filtered_df["Customer"].isin(customers)]

# --- METRICS ---
total_committed = filtered_df["Committed Orders"].sum()
total_achieved = filtered_df["Achieved Orders"].sum()
conversion_rate = (total_achieved / total_committed) * 100 if total_committed else 0
new_customers = filtered_df["New Customer"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 Total Committed Orders", f"${total_committed:,.0f}")
col2.metric("✅ Total Achieved Orders", f"${total_achieved:,.0f}")
col3.metric("🎯 Conversion Rate", f"{conversion_rate:.2f}%")
col4.metric("🆕 New Customers", f"{new_customers}")

# --- ORDERS COMPARISON BAR CHART ---
monthly_summary = (
    filtered_df.groupby(filtered_df["Month-Year"].dt.to_period("M"))[["Committed Orders", "Achieved Orders"]]
    .sum()
    .reset_index()
)
monthly_summary["Month-Year"] = monthly_summary["Month-Year"].dt.strftime("%b'%y")

fig = go.Figure()
fig.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Committed Orders"], name="Committed Orders", marker_color="#8ecae6", text=monthly_summary["Committed Orders"], textposition='outside'))
fig.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Achieved Orders"], name="Achieved Orders", marker_color="#219ebc", text=monthly_summary["Achieved Orders"], textposition='outside'))

fig.update_layout(title="📊 Monthly Orders Comparison (Committed vs Achieved)", xaxis_title="Month-Year", yaxis_title="Orders (USD)", barmode='group', bargap=0.25, template="plotly_white", legend=dict(title="", orientation="h", y=1.15, x=0.5, xanchor="center"), height=500)
fig.update_traces(marker_line_width=0.5, marker_line_color="gray")
st.plotly_chart(fig, use_container_width=True)

# --- TREEMAP ---
st.subheader("📘 Category-wise & Manager-wise Treemap")
fig_treemap = px.treemap(
    filtered_df,
    path=['Deal Manager', 'Plant Type', 'Customer'],
    values='Achieved Orders',
    color='Achieved Orders',
    color_continuous_scale='Blues',
    title='Category-wise Breakdown'
)
st.plotly_chart(fig_treemap, use_container_width=True)

# --- HEATMAP ---
st.subheader("🔥 Achieved Orders Heatmap (Manager × Month)")
heatmap_data = filtered_df.groupby(['Deal Manager', filtered_df['Month-Year'].dt.strftime('%b %Y')])['Achieved Orders'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month-Year', values='Achieved Orders').fillna(0)

fig_heatmap = px.imshow(
    heatmap_pivot,
    labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Orders"),
    x=heatmap_pivot.columns,
    y=heatmap_pivot.index,
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# --- CHOROPLETH MAP ---
st.subheader("🗺️ Region-wise Achieved Orders Map")
country_data = filtered_df.groupby('Country')['Achieved Orders'].sum().reset_index()
fig_choropleth = px.choropleth(
    country_data,
    locations='Country',
    locationmode='country names',
    color='Achieved Orders',
    color_continuous_scale='Blues',
    title='Achieved Orders by Country'
)
fig_choropleth.update_layout(geo=dict(showframe=False, showcoastlines=False))
st.plotly_chart(fig_choropleth, use_container_width=True)

# --- DOWNLOAD FILTERED DATA ---
st.subheader("⬇️ Download Filtered Data")

def convert_df(df, to_excel=True):
    if to_excel:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()
    else:
        return df.to_csv(index=False).encode('utf-8')

excel_data = convert_df(filtered_df, to_excel=True)
csv_data = convert_df(filtered_df, to_excel=False)

colx1, colx2 = st.columns(2)
with colx1:
    st.download_button(label="📥 Download as Excel", data=excel_data, file_name="filtered_orders.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with colx2:
    st.download_button(label="📥 Download as CSV", data=csv_data, file_name="filtered_orders.csv", mime="text/csv")
