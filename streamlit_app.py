import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from io import BytesIO

st.set_page_config(page_title="📊 Sales Dashboard", layout="wide")

# ==========================
# AUTHENTICATION AWARE UI
# ==========================
with st.sidebar:
    if hasattr(st, "experimental_user"):
        user_info = st.experimental_user
        st.success(f"🔒 Logged in as: {user_info['email']}")
    else:
        st.info("🔐 Sign in to access full dashboard.")

# ==========================
# CACHING DATA
# ==========================
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo/export?format=xlsx"
    df = pd.read_excel(url, engine='openpyxl')
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%Y')
    return df

df = load_data()

# ==========================
# SIDEBAR FILTERS
# ==========================
with st.sidebar:
    st.title("🔍 Filter Panel")
    with st.expander("📅 Month-Year"):
        selected_months = st.multiselect("", sorted(df['Month-Year'].dt.strftime('%b-%Y').unique()))
    with st.expander("👤 Deal Manager"):
        selected_deal_managers = st.multiselect("", df['Deal Manager'].unique())
    with st.expander("🏢 Customer"):
        selected_customers = st.multiselect("", df['Customer'].unique())
    with st.expander("🌍 Country"):
        selected_countries = st.multiselect("", df['Country'].unique())
    with st.expander("🏭 Plant Type"):
        selected_plants = st.multiselect("", df['Plant Type'].unique())

# ==========================
# APPLY FILTERS
# ==========================
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

# ==========================
# KPIs
# ==========================
st.title("📊 Founder Dashboard – Sales Performance")

kpi_cols = st.columns(3)

total_committed = filtered_df['Committed Revenue'].sum()
total_achieved = filtered_df['Achieved Revenue'].sum()
conversion = np.where(total_committed > 0, (total_achieved / total_committed) * 100, 0)

kpi_cols[0].metric("💰 Committed Revenue", f"${total_committed:,.0f}")
kpi_cols[1].metric("✅ Achieved Revenue", f"${total_achieved:,.0f}")
kpi_cols[2].metric("📈 Conversion %", f"{conversion:.1f}%", delta=f"{conversion - 100:+.1f}%", delta_color="normal")

# ==========================
# GROUPING
# ==========================
group_by = st.selectbox("📁 Group By", ['Month-Year', 'Deal Manager', 'Customer', 'Country', 'Plant Type'])

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
    grouped[group_by] = pd.to_datetime(grouped[group_by], errors='coerce')
    grouped = grouped.sort_values(by=group_by)
    grouped[group_by] = grouped[group_by].dt.strftime('%b-%Y')

# ==========================
# SAFE CALCULATIONS
# ==========================
grouped['Revenue Conversion %'] = np.where(grouped['Committed Revenue'] > 0,
                                           (grouped['Achieved Revenue'] / grouped['Committed Revenue']) * 100, 0)
grouped['Orders Conversion %'] = np.where(grouped['Committed Orders'] > 0,
                                          (grouped['Achieved Orders'] / grouped['Committed Orders']) * 100, 0)
grouped['GM Conversion %'] = np.where(grouped['Committed Gross Margin'] > 0,
                                      (grouped['Achieved Gross Margin'] / grouped['Committed Gross Margin']) * 100, 0)

# ==========================
# DATA TABLE
# ==========================
with st.expander("📋 View Aggregated Data"):
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
    }), use_container_width=True)

# ==========================
# DOWNLOAD BUTTON
# ==========================
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Summary')
        writer.save()
    return output.getvalue()

st.download_button("⬇️ Download Filtered Data", convert_df_to_excel(grouped),
                   file_name="sales_summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==========================
# CHART FUNCTION
# ==========================
def plot_chart(title, df, x, y, colors, yaxis_title):
    try:
        if len(df) > 20:
            df = df.sort_values(by=y[1] if len(y) > 1 else y[0], ascending=False).head(20)
        fig = px.bar(df, x=x, y=y, barmode='group', text_auto='.2s', color_discrete_sequence=colors)
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

# ==========================
# VISUALIZATIONS
# ==========================
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"💰 Revenue by {group_by}")
        plot_chart(f"Revenue by {group_by}", grouped, group_by,
                   ['Committed Revenue', 'Achieved Revenue'],
                   ['#1f77b4', '#2ca02c'], "Revenue (USD)")

    with col2:
        st.subheader(f"📦 Orders by {group_by}")
        plot_chart(f"Orders by {group_by}", grouped, group_by,
                   ['Committed Orders', 'Achieved Orders'],
                   ['#ff7f0e', '#9467bd'], "Orders")

st.subheader(f"📈 Gross Margin by {group_by}")
plot_chart(f"Gross Margin by {group_by}", grouped, group_by,
           ['Committed Gross Margin', 'Achieved Gross Margin'],
           ['#d62728', '#17becf'], "Gross Margin (USD)")
