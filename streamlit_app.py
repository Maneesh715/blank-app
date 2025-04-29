import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Worldref Dashboard", layout="wide")

# --- NAVIGATION MENU ---
st.sidebar.title("📁 Navigation")
page = st.sidebar.selectbox("Go to", ["📊 Orders Dashboard", "📈 Sheet2 Dashboard"])

if page == "📊 Orders Dashboard":
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

# --- DATA CLEANING & PROCESSING ---
df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y")
df["New Customer"] = df["New Customer"].fillna(0).astype(int)
df["Committed Orders"] = pd.to_numeric(df["Committed Orders"], errors='coerce').fillna(0)
df["Achieved Orders"] = pd.to_numeric(df["Achieved Orders"], errors='coerce').fillna(0)
df["Conversion Rate (%)"] = df.apply(lambda row: (row["Achieved Orders"] / row["Committed Orders"] * 100) if row["Committed Orders"] else 0, axis=1)

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔎 Filters")
deal_managers = st.sidebar.multiselect("Select Deal Manager(s):", options=sorted(df["Deal Manager"].dropna().unique()))
countries = st.sidebar.multiselect("Select Country(ies):", options=sorted(df["Country"].dropna().unique()))
plants = st.sidebar.multiselect("Select Plant Type(s):", options=sorted(df["Plant Type"].dropna().unique()))
customers = st.sidebar.multiselect("Select Customer(s):", options=sorted(df["Customer"].dropna().unique()))

filtered_df = df.copy()
if deal_managers:
    filtered_df = filtered_df[filtered_df["Deal Manager"].isin(deal_managers)]
if countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
if plants:
    filtered_df = filtered_df[filtered_df["Plant Type"].isin(plants)]
if customers:
    filtered_df = filtered_df[filtered_df["Customer"].isin(customers)]

# --- SUMMARY METRICS ---
total_committed = filtered_df["Committed Orders"].sum()
total_achieved = filtered_df["Achieved Orders"].sum()
conversion_rate = (total_achieved / total_committed) * 100 if total_committed else 0
new_customers = filtered_df["New Customer"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 Total Committed Orders", f"${total_committed:,.0f}")
col2.metric("✅ Total Achieved Orders", f"${total_achieved:,.0f}")
col3.metric("🎯 Conversion Rate", f"{conversion_rate:.2f}%")
col4.metric("🆕 New Customers", f"{new_customers}")

# --- ORDERS COMPARISON ---
monthly_summary = (
    filtered_df.groupby(filtered_df["Month-Year"].dt.to_period("M"))[["Committed Orders", "Achieved Orders"]]
    .sum()
    .reset_index()
)
monthly_summary["Month-Year"] = monthly_summary["Month-Year"].dt.strftime("%b'%y")

fig_orders = go.Figure()
fig_orders.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Committed Orders"], name="Committed Orders", marker_color="#66c2a5", text=monthly_summary["Committed Orders"], textposition='outside'))
fig_orders.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Achieved Orders"], name="Achieved Orders", marker_color="#1d3557", text=monthly_summary["Achieved Orders"], textposition='outside'))
fig_orders.update_layout(title="📊 Monthly Orders Comparison", xaxis_title="Month-Year", yaxis_title="Orders (USD)", barmode='group', template="plotly_white", legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"))
st.plotly_chart(fig_orders, use_container_width=True)

# --- CATEGORY-WISE & MANAGER-WISE TREEMAP ---
st.subheader("📘 Category-wise & Manager-wise Breakdown (Treemap)")
st.markdown("""
This treemap shows:
- **Deal Managers** managing the highest orders.
- **Plant Types** contributing to revenue.
- Top **Customers** within each plant type.
""")
fig_treemap = px.treemap(
    filtered_df,
    path=['Deal Manager', 'Plant Type', 'Customer'],
    values='Achieved Orders',
    color='Achieved Orders',
    color_continuous_scale='Viridis',
    title='Category-wise & Manager-wise Breakdown',
    hover_data={'Achieved Orders': ':,.0f'}
)
fig_treemap.update_traces(root_color="lightgrey")
st.plotly_chart(fig_treemap, use_container_width=True)

# --- HEATMAP OF ACHIEVED ORDERS ---
st.subheader("🔥 Achieved Orders Heatmap (Manager × Month)")
heatmap_data = filtered_df.groupby(['Deal Manager', filtered_df['Month-Year'].dt.strftime('%b %Y')])['Achieved Orders'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month-Year', values='Achieved Orders').fillna(0)

fig_heatmap = px.imshow(
    heatmap_pivot,
    labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Orders"),
    color_continuous_scale='Turbo',
    aspect="auto",
    text_auto=True
)
fig_heatmap.update_layout(title='Achieved Orders by Manager & Month', xaxis_side="top")
st.plotly_chart(fig_heatmap, use_container_width=True)

# --- REGION-WISE ACHIEVED ORDERS MAP ---
st.subheader("🗺️ Region-wise Achieved Orders")
country_data = filtered_df.groupby('Country')['Achieved Orders'].sum().reset_index()

fig_choropleth = px.choropleth(
    country_data,
    locations='Country',
    locationmode='country names',
    color='Achieved Orders',
    color_continuous_scale='Plasma',
    title='Achieved Orders by Country',
    hover_name='Country'
)
fig_choropleth.update_geos(showframe=True, showcoastlines=True, projection_type="natural earth")
fig_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
st.plotly_chart(fig_choropleth, use_container_width=True)

# --- DOWNLOAD FILTERED DATA ---
st.subheader("⬇️ Download Filtered Data")

def convert_df(df, to_excel=True):
    output = BytesIO()
    if to_excel:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()
    else:
        return df.to_csv(index=False).encode('utf-8')

excel_data = convert_df(filtered_df, to_excel=True)
csv_data = convert_df(filtered_df, to_excel=False)

colx1, colx2 = st.columns(2)
with colx1:
    st.download_button("📥 Download as Excel", data=excel_data, file_name="filtered_orders.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with colx2:
    st.download_button("📥 Download as CSV", data=csv_data, file_name="filtered_orders.csv", mime="text/csv")



