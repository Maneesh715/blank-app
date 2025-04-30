import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Worldref Dashboard", layout="wide")

st.sidebar.title("📁 Navigation")
page = st.sidebar.selectbox("Go to", ["📊 Orders Dashboard", "📊 Revenue Dashboard", "📊 Gross Margin Dashboard"])

if page == "📊 Orders Dashboard":
    SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    SHEET_NAME = "Sheet1"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data(CSV_URL)

    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y")
    df["New Customer"] = df["New Customer"].fillna(0).astype(int)
    df["Committed Orders"] = pd.to_numeric(df["Committed Orders"], errors='coerce').fillna(0)
    df["Achieved Orders"] = pd.to_numeric(df["Achieved Orders"], errors='coerce').fillna(0)
    df["Conversion Rate (%)"] = df.apply(lambda row: (row["Achieved Orders"] / row["Committed Orders"] * 100) if row["Committed Orders"] else 0, axis=1)

    st.sidebar.header("🔎 Filters")
    month_year = st.sidebar.multiselect("Select Month-Year(s):", options=sorted(df["Month-Year"].dropna().unique()))
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

    total_committed = filtered_df["Committed Orders"].sum()
    total_achieved = filtered_df["Achieved Orders"].sum()
    conversion_rate = (total_achieved / total_committed) * 100 if total_committed else 0
    new_customers = filtered_df["New Customer"].sum()
    average_order_size = (total_achieved / len(filtered_df)) if len(filtered_df) > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📌 Total Committed Orders", f"${total_committed:,.0f}")
    col2.metric("✅ Total Achieved Orders", f"${total_achieved:,.0f}")
    col3.metric("🎯 Conversion Rate", f"{conversion_rate:.2f}%")
    col4.metric("🆕 New Customers", f"{new_customers}")
    col5.metric("📦 Avg. Order Size", f"${average_order_size:,.0f}")

    # --- Monthly Orders Comparison ---
    monthly_summary = (
        filtered_df.groupby(filtered_df["Month-Year"].dt.to_period("M"))[["Committed Orders", "Achieved Orders"]]
        .sum()
        .reset_index()
    )
    monthly_summary["Month-Year"] = monthly_summary["Month-Year"].dt.strftime("%b'%y")
    monthly_summary["Conversion Rate (%)"] = monthly_summary.apply(
        lambda row: (row["Achieved Orders"] / row["Committed Orders"] * 100) if row["Committed Orders"] else 0,
        axis=1
    )

    fig_orders = make_subplots(specs=[[{"secondary_y": True}]])
    fig_orders.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Committed Orders"],
                                name="Committed Orders", marker_color="#66c2a5", text=monthly_summary["Committed Orders"], textposition='outside'),
                         secondary_y=False)
    fig_orders.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Achieved Orders"],
                                name="Achieved Orders", marker_color="#1d3557", text=monthly_summary["Achieved Orders"], textposition='outside'),
                         secondary_y=False)
    fig_orders.add_trace(go.Scatter(x=monthly_summary["Month-Year"], y=monthly_summary["Conversion Rate (%)"],
                                    name="Conversion Rate (%)", mode='lines+markers', line=dict(color="#e76f51", width=3), marker=dict(size=6)),
                         secondary_y=True)

    fig_orders.update_layout(
        title="📊 Monthly Orders & Conversion Rate",
        xaxis_title="Month-Year",
        yaxis_title="Orders (USD)",
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        template="plotly_white",
        barmode='group',
        height=500
    )
    fig_orders.update_yaxes(title_text="Orders (USD)", secondary_y=False)
    fig_orders.update_yaxes(title_text="Conversion Rate (%)", secondary_y=True)

    st.plotly_chart(fig_orders, use_container_width=True)

    # --- Treemap with Drill-down ---
    st.subheader("📘 Category-wise & Manager-wise Breakdown (Treemap)")
    fig_treemap = px.treemap(
        filtered_df,
        path=['Deal Manager', 'Plant Type', 'Customer'],
        values='Achieved Orders',
        color='Achieved Orders',
        color_continuous_scale='Viridis',
        custom_data=['Deal Manager', 'Plant Type', 'Customer', 'Achieved Orders'],
        title='Category-wise & Manager-wise Breakdown'
    )
    fig_treemap.update_traces(root_color="lightgrey")
    st.plotly_chart(fig_treemap, use_container_width=True)

    #selected_treemap = st.plotly_chart(fig_treemap, use_container_width=True)
    #st.info("🖱️ Click a Treemap section to drill down — feature for future interactivity.")

    # --- Heatmap with Drill-down ---
    st.subheader("🔥 Achieved Orders Heatmap (Manager × Month)")
    heatmap_data = filtered_df.groupby(['Deal Manager', filtered_df['Month-Year'].dt.strftime('%b %Y')])['Achieved Orders'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month-Year', values='Achieved Orders').fillna(0)

    heatmap_pivot.loc['Average'] = heatmap_pivot.mean()
    heatmap_pivot['Average'] = heatmap_pivot.mean(axis=1)

    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Orders"),
        color_continuous_scale='Turbo',
        aspect="auto",
        text_auto=True
    )
    fig_heatmap.update_layout(
        title='Achieved Orders by Manager & Month (with Averages)',
        xaxis_side="top"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    #st.info("🖱️ Click a heatmap cell to drill down — full interactivity can be added via `plotly_click` callbacks.")

    # --- Map ---
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

    # --- Download ---
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

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name="filtered_orders_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="filtered_orders_data.csv",
            mime="text/csv"
        )

elif page == "📊 Revenue Dashboard":
    SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    SHEET_NAME = "Sheet2"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data(CSV_URL)

    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y", errors='coerce')
    df["New Customer"] = df["New Customer"].fillna(0).astype(int)
    df["Committed Revenue"] = pd.to_numeric(df["Committed Revenue"], errors='coerce').fillna(0)
    df["Achieved Revenue"] = pd.to_numeric(df["Achieved Revenue"], errors='coerce').fillna(0)
    df["Conversion Rate (%)"] = df.apply(lambda row: (row["Achieved Revenue"] / row["Committed Revenue"] * 100) if row["Committed Revenue"] else 0, axis=1)

    st.sidebar.header("🔎 Filters")
    month_year = st.sidebar.multiselect("Select Month-Year(s):", options=sorted(df["Month-Year"].dropna().unique()))
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

    total_committed = filtered_df["Committed Revenue"].sum()
    total_achieved = filtered_df["Achieved Revenue"].sum()
    conversion_rate = (total_achieved / total_committed) * 100 if total_committed else 0
    new_customers = filtered_df["New Customer"].sum()
    average_revenue_size = (total_achieved / len(filtered_df)) if len(filtered_df) > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📌 Total Committed Revenue", f"${total_committed:,.0f}")
    col2.metric("✅ Total Achieved Revenue", f"${total_achieved:,.0f}")
    col3.metric("🎯 Conversion Rate", f"{conversion_rate:.2f}%")
    col4.metric("🆕 New Customers", f"{new_customers}")
    col5.metric("📦 Avg. Revenue Size", f"${average_revenue_size:,.0f}")

    # --- Monthly Orders Comparison ---
    monthly_summary = (
        filtered_df.groupby(filtered_df["Month-Year"].dt.to_period("M"))[["Committed Revenue", "Achieved Revenue"]]
        .sum()
        .reset_index()
    )
    monthly_summary["Month-Year"] = monthly_summary["Month-Year"].dt.strftime("%b'%y")
    monthly_summary["Conversion Rate (%)"] = monthly_summary.apply(
        lambda row: (row["Achieved Revenue"] / row["Committed Revenue"] * 100) if row["Committed Revenue"] else 0,
        axis=1
    )

    fig_revenue = make_subplots(specs=[[{"secondary_y": True}]])
    fig_revenue.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Committed Revenue"],
                                name="Committed Revenue", marker_color="#66c2a5", text=monthly_summary["Committed Revenue"], textposition='outside'),
                         secondary_y=False)
    fig_revenue.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Achieved Revenue"],
                                name="Achieved Revenue", marker_color="#1d3557", text=monthly_summary["Achieved Revenue"], textposition='outside'),
                         secondary_y=False)
    fig_revenue.add_trace(go.Scatter(x=monthly_summary["Month-Year"], y=monthly_summary["Conversion Rate (%)"],
                                    name="Conversion Rate (%)", mode='lines+markers', line=dict(color="#e76f51", width=3), marker=dict(size=6)),
                         secondary_y=True)

    fig_revenue.update_layout(
        title="📊 Monthly Revenue & Conversion Rate",
        xaxis_title="Month-Year",
        yaxis_title="Revenue (USD)",
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        template="plotly_white",
        barmode='group',
        height=500
    )
    fig_revenue.update_yaxes(title_text="Revenue (USD)", secondary_y=False)
    fig_revenue.update_yaxes(title_text="Conversion Rate (%)", secondary_y=True)

    st.plotly_chart(fig_revenue, use_container_width=True)

    # --- Treemap with Drill-down ---
    st.subheader("📘 Category-wise & Manager-wise Breakdown (Treemap)")
    fig_treemap = px.treemap(
        filtered_df,
        path=['Deal Manager', 'Plant Type', 'Customer'],
        values='Achieved Revenue',
        color='Achieved Revenue',
        color_continuous_scale='Viridis',
        custom_data=['Deal Manager', 'Plant Type', 'Customer', 'Achieved Revenue'],
        title='Category-wise & Manager-wise Breakdown'
    )
    fig_treemap.update_traces(root_color="lightgrey")
    st.plotly_chart(fig_treemap, use_container_width=True)

    #selected_treemap = st.plotly_chart(fig_treemap, use_container_width=True)
    #st.info("🖱️ Click a Treemap section to drill down — feature for future interactivity.")

    # --- Heatmap with Drill-down ---
    st.subheader("🔥 Achieved Revenue Heatmap (Manager × Month)")
    heatmap_data = filtered_df.groupby(['Deal Manager', filtered_df['Month-Year'].dt.strftime('%b %Y')])['Achieved Revenue'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month-Year', values='Achieved Revenue').fillna(0)

    heatmap_pivot.loc['Average'] = heatmap_pivot.mean()
    heatmap_pivot['Average'] = heatmap_pivot.mean(axis=1)

    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Revenue"),
        color_continuous_scale='Turbo',
        aspect="auto",
        text_auto=True
    )
    fig_heatmap.update_layout(
        title='Achieved Revenue by Manager & Month (with Averages)',
        xaxis_side="top"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    #st.info("🖱️ Click a heatmap cell to drill down — full interactivity can be added via `plotly_click` callbacks.")

    # --- Map ---
    st.subheader("🗺️ Region-wise Achieved Revenue")
    country_data = filtered_df.groupby('Country')['Achieved Revenue'].sum().reset_index()
    fig_choropleth = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Achieved Revenue',
        color_continuous_scale='Plasma',
        title='Achieved Revenue by Country',
        hover_name='Country'
    )
    fig_choropleth.update_geos(showframe=True, showcoastlines=True, projection_type="natural earth")
    fig_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig_choropleth, use_container_width=True)

    # --- Download ---
    st.subheader("⬇️ Download Filtered Data")

    def convert_df(df, to_excel=True):
        output = BytesIO()
        if to_excel:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet2')
            return output.getvalue()
        else:
            return df.to_csv(index=False).encode('utf-8')

    excel_data = convert_df(filtered_df, to_excel=True)
    csv_data = convert_df(filtered_df, to_excel=False)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name="filtered_revenue_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="filtered_revenue_data.csv",
            mime="text/csv"
        )

else page == "📊 Gross Margin Dashboard":
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    # ------------------ SETTINGS ------------------
    st.set_page_config(page_title="Gross Margin Dashboard", layout="wide")
    st.title("📊 Gross Margin Dashboard")

    # ------------------ DATA LOADING ------------------
    SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    SHEET_NAME = "Sheet3"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data(CSV_URL)

    # ------------------ PREPROCESSING ------------------
    usd_conversion = 86
    value_cols_inr = [
        "Committed COGS", "Achieved COGS", "Committed Logistics", "Achieved Logistics",
        "Committed P&F", "Achieved P&F", "Committed Associate Payment", "Achieved Associate Payment",
        "Committed Revenue", "Achieved Revenue"
    ]

    for col in value_cols_inr:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col + " (USD)"] = df[col] / usd_conversion

    # Compute gross margins (in USD)
    df["Committed Gross Margin (USD)"] = df["Committed Revenue (USD)"] - (
        df["Committed COGS (USD)"] + df["Committed Logistics (USD)"] + df["Committed P&F (USD)"] + df["Committed Associate Payment (USD)"])

    df["Achieved Gross Margin (USD)"] = df["Achieved Revenue (USD)"] - (
        df["Achieved COGS (USD)"] + df["Achieved Logistics (USD)"] + df["Achieved P&F (USD)"] + df["Achieved Associate Payment (USD)"])

    # Compute % values using total-based method (not simple mean)
    df["Committed Gross Margin (%)"] = np.where(df["Committed Revenue (USD)"] > 0,
        (df["Committed Gross Margin (USD)"] / df["Committed Revenue (USD)"]) * 100, np.nan)
    df["Achieved Gross Margin (%)"] = np.where(df["Achieved Revenue (USD)"] > 0,
        (df["Achieved Gross Margin (USD)"] / df["Achieved Revenue (USD)"]) * 100, np.nan)
    df["Margin Realization (%)"] = np.where(df["Committed Gross Margin (USD)"] > 0,
        (df["Achieved Gross Margin (USD)"] / df["Committed Gross Margin (USD)"]) * 100, np.nan)

    # ------------------ VISUALIZATIONS ------------------
    st.header("📅 Monthly Trends")
    monthly = df.groupby("Month-Year").agg({
        "Committed Gross Margin (USD)": "sum",
        "Achieved Gross Margin (USD)": "sum",
        "Committed Revenue (USD)": "sum",
        "Achieved Revenue (USD)": "sum"
    }).reset_index()

    monthly["Committed Gross Margin (%)"] = (monthly["Committed Gross Margin (USD)"] / monthly["Committed Revenue (USD)"]) * 100
    monthly["Achieved Gross Margin (%)"] = (monthly["Achieved Gross Margin (USD)"] / monthly["Achieved Revenue (USD)"]) * 100
    monthly["Margin Realization (%)"] = (monthly["Achieved Gross Margin (USD)"] / monthly["Committed Gross Margin (USD)"]) * 100

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=monthly["Month-Year"], y=monthly["Committed Gross Margin (USD)"], name="Committed GM (USD)", marker_color='lightblue'))
    fig1.add_trace(go.Bar(x=monthly["Month-Year"], y=monthly["Achieved Gross Margin (USD)"], name="Achieved GM (USD)", marker_color='green'))
    fig1.add_trace(go.Scatter(x=monthly["Month-Year"], y=monthly["Margin Realization (%)"], name="Margin Realization (%)", yaxis="y2", mode="lines+markers", line=dict(color="black")))

    fig1.update_layout(
        title="Monthly Gross Margin and Margin Realization",
        xaxis_title="Month-Year",
        yaxis=dict(title="Gross Margin (USD)"),
        yaxis2=dict(title="Margin Realization (%)", overlaying="y", side="right", range=[0, 120]),
        barmode='group'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------ TREEMAP: Category-wise & Manager-wise ------------------
    st.header("📂 Gross Margin Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.treemap(df, path=['Plant Type', 'Deal Manager'],
            values='Achieved Gross Margin (USD)',
            title="Achieved Gross Margin (USD) by Category & Manager")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        df_valid = df[df["Achieved Revenue (USD)"] > 0]
        df_valid["Achieved Gross Margin (%)"] = (df_valid["Achieved Gross Margin (USD)"] / df_valid["Achieved Revenue (USD)"]) * 100
        fig3 = px.treemap(df_valid, path=['Plant Type', 'Deal Manager'],
            values='Achieved Gross Margin (%)',
            title="Achieved Gross Margin (%) by Category & Manager")
        st.plotly_chart(fig3, use_container_width=True)

    # ------------------ HEATMAP: Manager x Month ------------------
    st.header("🌡️ Achieved Gross Margin Heatmap")
    heat_df = df.groupby(["Deal Manager", "Month-Year"]).agg({
        "Achieved Gross Margin (USD)": "sum",
        "Achieved Revenue (USD)": "sum"
    }).reset_index()
    heat_df["Achieved Gross Margin (%)"] = (heat_df["Achieved Gross Margin (USD)"] / heat_df["Achieved Revenue (USD)"]) * 100

    pivot_usd = heat_df.pivot(index="Deal Manager", columns="Month-Year", values="Achieved Gross Margin (USD)")
    pivot_pct = heat_df.pivot(index="Deal Manager", columns="Month-Year", values="Achieved Gross Margin (%)")

    st.subheader("💰 Achieved Gross Margin (USD)")
    st.dataframe(pivot_usd.style.format("{:,.0f}"))

    st.subheader("📈 Achieved Gross Margin (%)")
    st.dataframe(pivot_pct.style.format("{:.1f}%"))

    # ------------------ COUNTRY-WISE ANALYSIS ------------------
    st.header("🌍 Region-wise Analysis")
    country_df = df.groupby("Country").agg({
        "Achieved Gross Margin (USD)": "sum",
        "Achieved Revenue (USD)": "sum"
    }).reset_index()
    country_df["Achieved Gross Margin (%)"] = (country_df["Achieved Gross Margin (USD)"] / country_df["Achieved Revenue (USD)"]) * 100

    fig4 = px.bar(country_df.sort_values("Achieved Gross Margin (USD)", ascending=False),
                  x="Country", y="Achieved Gross Margin (USD)",
                  title="Achieved Gross Margin (USD) by Country")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.bar(country_df.sort_values("Achieved Gross Margin (%)", ascending=False),
                  x="Country", y="Achieved Gross Margin (%)",
                  title="Achieved Gross Margin (%) by Country")
    st.plotly_chart(fig5, use_container_width=True)

    # ------------------ END ------------------

    # --- Download ---
    st.subheader("⬇️ Download Filtered Data")

    def convert_df(df, to_excel=True):
        output = BytesIO()
        if to_excel:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet3')
            return output.getvalue()
        else:
            return df.to_csv(index=False).encode('utf-8')

    excel_data = convert_df(filtered_df, to_excel=True)
    csv_data = convert_df(filtered_df, to_excel=False)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download as Excel",
            data=excel_data,
            file_name="filtered_GM_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="filtered_GM_data.csv",
            mime="text/csv"
        )
