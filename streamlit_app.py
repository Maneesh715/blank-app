import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Orders Dashboard", layout="wide")

# --- TITLE ---
st.title("📊 Orders Comparison Dashboard (Committed vs Achieved)")

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

# --- FILTER SIDEBAR ---
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

# --- ORDERS COMPARISON CHART (TIME SERIES) ---
monthly_summary = (
    filtered_df.groupby(filtered_df["Month-Year"].dt.to_period("M"))[["Committed Orders", "Achieved Orders"]]
    .sum()
    .reset_index()
)
monthly_summary["Month-Year"] = monthly_summary["Month-Year"].astype(str)

fig = px.bar(
    monthly_summary.melt(id_vars="Month-Year", value_vars=["Committed Orders", "Achieved Orders"]),
    x="Month-Year",
    y="value",
    color="variable",
    barmode="group",
    labels={"value": "USD ($)", "Month-Year": "Month"},
    title="📆 Monthly Orders: Committed vs Achieved"
)
st.plotly_chart(fig, use_container_width=True)

# --- DOWNLOAD DATA ---
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
    st.download_button(
        label="📥 Download as Excel",
        data=excel_data,
        file_name="filtered_orders.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
with colx2:
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="filtered_orders.csv",
        mime="text/csv"
    )
