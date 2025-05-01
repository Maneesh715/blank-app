import streamlit as st
import pandas as pd
import numpy as np
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

else:
    # ------------------ SETTINGS ------------------
    #st.set_page_config(page_title="Gross Margin Dashboard", layout="wide")
    st.title("📊 Gross Margin Dashboard")

    # ------------------ DATA LOADING ------------------
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    SHEET_NAME = "Sheet3"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data(CSV_URL)

    # Preprocessing
    usd_conversion = 86
    value_cols_inr = [
        "Committed COGS", "Achieved COGS", "Committed Logistics", "Achieved Logistics",
        "Committed P&F", "Achieved P&F", "Committed Associate Payment", "Achieved Associate Payment",
        "Committed Revenue", "Achieved Revenue"
    ]

    for col in value_cols_inr:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col + " (USD)"] = df[col] / usd_conversion

    df["Committed Gross Margin (USD)"] = df["Committed Revenue (USD)"] - (
        df["Committed COGS (USD)"] + df["Committed Logistics (USD)"] + df["Committed P&F (USD)"] + df["Committed Associate Payment (USD)"])

    df["Achieved Gross Margin (USD)"] = df["Achieved Revenue (USD)"] - (
        df["Achieved COGS (USD)"] + df["Achieved Logistics (USD)"] + df["Achieved P&F (USD)"] + df["Achieved Associate Payment (USD)"])

    df["Committed Gross Margin (%)"] = np.where(df["Committed Revenue (USD)"] > 0,
        (df["Committed Gross Margin (USD)"] / df["Committed Revenue (USD)"]) * 100, np.nan)
    df["Achieved Gross Margin (%)"] = np.where(df["Achieved Revenue (USD)"] > 0,
        (df["Achieved Gross Margin (USD)"] / df["Achieved Revenue (USD)"]) * 100, np.nan)
    df["Margin Realization (%)"] = np.where(df["Committed Gross Margin (USD)"] > 0,
        (df["Achieved Gross Margin (USD)"] / df["Committed Gross Margin (USD)"]) * 100, np.nan)

    # Display KPI cards in 3 columns
    col1, col2, col3 = st.columns(3)

    col1.metric("Achieved Gross Margin (USD)", f"${Achieved Gross Margin (USD):,.2f}")
    col2.metric("Achieved Gross Margin (%)", f"{Achieved Gross Margin (%):.2f}%")
    col3.metric("Margin Realization (%)", f"{Margin Realization (%):.2f}%")


    # ------------------ SIDEBAR FILTERS ------------------
    st.sidebar.header("🔎 Filters")
    month_year = st.sidebar.multiselect("Select Month-Year(s):", options=sorted(df["Month-Year"].dropna().unique()))
    deal_managers = st.sidebar.multiselect("Select Deal Manager(s):", options=sorted(df["Deal Manager"].dropna().unique()))
    countries = st.sidebar.multiselect("Select Country(ies):", options=sorted(df["Country"].dropna().unique()))
    plants = st.sidebar.multiselect("Select Plant Type(s):", options=sorted(df["Plant Type"].dropna().unique()))
    customers = st.sidebar.multiselect("Select Customer(s):", options=sorted(df["Customer"].dropna().unique()))

    # Apply filters
    filtered_df = df.copy()
    if month_year:
        filtered_df = filtered_df[filtered_df["Month-Year"].isin(month_year)]
    if deal_managers:
        filtered_df = filtered_df[filtered_df["Deal Manager"].isin(deal_managers)]
    if countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
    if plants:
        filtered_df = filtered_df[filtered_df["Plant Type"].isin(plants)]
    if customers:
        filtered_df = filtered_df[filtered_df["Customer"].isin(customers)]

    # ------------------ VISUALIZATION ------------------
    st.header("📅 Monthly Gross Margin & Margin Realization")

    # Add sorting column
    filtered_df["MonthYearSort"] = pd.to_datetime(filtered_df["Month-Year"], format="%b %Y", errors='coerce')

    # Filter out rows with zero Achieved Revenue (USD)
    filtered_df = filtered_df[filtered_df["Achieved Revenue (USD)"] != 0]

    # Aggregate
    monthly = (
        filtered_df.groupby("MonthYearSort")
        .agg({
            "Committed Gross Margin (USD)": "sum",
            "Achieved Gross Margin (USD)": "sum",
            "Committed Revenue (USD)": "sum",
            "Achieved Revenue (USD)": "sum"
        })
        .reset_index()
    )

    monthly["Month-Year"] = monthly["MonthYearSort"].dt.strftime("%b %Y")

    # Calculate %
    monthly["Committed Gross Margin (%)"] = np.where(
        monthly["Committed Revenue (USD)"] != 0,
        (monthly["Committed Gross Margin (USD)"] / monthly["Committed Revenue (USD)"]) * 100,
        0
    )

    monthly["Achieved Gross Margin (%)"] = np.where(
        monthly["Achieved Revenue (USD)"] != 0,
        (monthly["Achieved Gross Margin (USD)"] / monthly["Achieved Revenue (USD)"]) * 100,
        0
    )

    monthly["Margin Realization (%)"] = np.where(
        (monthly["Committed Gross Margin (USD)"] != 0) & (monthly["Achieved Gross Margin (USD)"] != 0),
        (monthly["Achieved Gross Margin (USD)"] / monthly["Committed Gross Margin (USD)"]) * 100,
        0
    )

    # Plot
    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        x=monthly["Month-Year"],
        y=monthly["Committed Gross Margin (USD)"],
        name="Committed GM (USD)",
        marker_color='lightblue',
        hovertemplate=(
            "Month: %{x}<br>"
            "Committed GM: $%{y:,.0f}<br>"
            "Committed GM (%): %{customdata:.1f}%"
        ),
        customdata=monthly[["Committed Gross Margin (%)"]].values
    ))

    fig1.add_trace(go.Bar(
        x=monthly["Month-Year"],
        y=monthly["Achieved Gross Margin (USD)"],
        name="Achieved GM (USD)",
        marker_color='green',
        hovertemplate=(
            "Month: %{x}<br>"
            "Achieved GM: $%{y:,.0f}<br>"
            "Achieved GM (%): %{customdata:.1f}%"
        ),
        customdata=monthly[["Achieved Gross Margin (%)"]].values
    ))

    fig1.add_trace(go.Scatter(
        x=monthly["Month-Year"],
        y=monthly["Margin Realization (%)"],
        name="Margin Realization (%)",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="black"),
        hovertemplate=(
            "Month: %{x}<br>"
            "Margin Realization: %{y:.1f}%"
        )
    ))

    fig1.update_layout(
        title="Monthly Gross Margin and Margin Realization",
        xaxis_title="Month-Year",
        yaxis=dict(title="Gross Margin (USD)"),
        yaxis2=dict(title="Margin Realization (%)", overlaying="y", side="right", range=[0, 120]),
        barmode='group'
    )

    st.plotly_chart(fig1, use_container_width=True)


    # ------------------ TREEMAP: Category-wise & Manager-wise ------------------
    st.subheader("📘 Category-wise & Manager-wise Breakdown (Treemap)")

    # ✅ Step 1: Ensure Achieved Gross Margin (%) is calculated
    if 'Achieved Gross Margin (%)' not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['Achieved Revenue'] = filtered_df['Achieved Revenue'].replace(0, np.nan)
        filtered_df['Achieved Gross Margin (%)'] = (
            (filtered_df['Achieved Revenue'] - (filtered_df['Achieved COGS'] + filtered_df['Achieved Logistics'] + filtered_df['Achieved P&F'] + filtered_df['Achieved Associate Payment'])) / filtered_df['Achieved Revenue']
        ) * 100

    # ✅ Step 2: Fill missing/NaN values to ensure all nodes are included
    filtered_df['Achieved Gross Margin (%)'] = filtered_df['Achieved Gross Margin (%)'].fillna(0)

    # ✅ Step 3: Replace zeros with a small positive value to make them visible in the treemap
    filtered_df['Achieved Gross Margin Display'] = filtered_df['Achieved Gross Margin (%)'].apply(lambda x: x if x > 0 else 0.01)

    # ✅ Step 4: Create the treemap
    fig_treemap = px.treemap(
        filtered_df,
        path=['Deal Manager', 'Plant Type', 'Customer'],
        values='Achieved Gross Margin Display',
        color='Achieved Gross Margin (%)',
        color_continuous_scale='Viridis',
        custom_data=['Deal Manager', 'Plant Type', 'Customer', 'Achieved Gross Margin (%)'],
        title='Category-wise & Manager-wise Breakdown'
    )

    # ✅ Step 5: Update visuals for clarity
    fig_treemap.update_traces(
        root_color="lightgrey",
        hovertemplate='<b>%{label}</b><br>Gross Margin: %{customdata[3]:.2f}%',
        texttemplate='%{label}<br>%{customdata[3]:.1f}%',
        textinfo='label+value'
    )

    # ✅ Step 6: Display in Streamlit
    st.plotly_chart(fig_treemap, use_container_width=True)

    # ------------------ HEATMAP: Manager x Month ------------------
    st.subheader("🔥 Achieved Gross Margin (%) Heatmap (Manager × Month)")

    # Step 1: Filter rows where Achieved Revenue is not zero
    filtered_df = df[df['Achieved Revenue'] != 0].copy()

    # Step 2: Create 'MonthYearSort' from 'Month-Year' for sorting
    filtered_df['MonthYearSort'] = pd.to_datetime(filtered_df['Month-Year'], format='%b %Y')

    # Step 3: Aggregate raw financials by Deal Manager and MonthYearSort
    heatmap_data = (
        filtered_df.groupby(['Deal Manager', 'MonthYearSort'])
        .agg({
            'Achieved Revenue': 'sum',
            'Achieved COGS': 'sum',
            'Achieved Logistics': 'sum',
            'Achieved P&F': 'sum',
            'Achieved Associate Payment': 'sum'
        })
        .reset_index()
    )

    # Step 4: Compute Achieved Gross Margin (%) from aggregated values
    heatmap_data['Achieved Gross Margin (%)'] = (
        (heatmap_data['Achieved Revenue'] - (
            heatmap_data['Achieved COGS'] +
            heatmap_data['Achieved Logistics'] +
            heatmap_data['Achieved P&F'] +
            heatmap_data['Achieved Associate Payment']
        )) / heatmap_data['Achieved Revenue']
    ) * 100

    # Step 5: Format 'Month-Year' for display
    heatmap_data['Month-Year'] = heatmap_data['MonthYearSort'].dt.strftime('%b %Y')

    # Step 6: Pivot data for heatmap
    heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month-Year', values='Achieved Gross Margin (%)').fillna(0)

    # Step 7: Sort columns chronologically
    sorted_columns = sorted(heatmap_pivot.columns, key=lambda x: pd.to_datetime(x, format='%b %Y'))
    heatmap_pivot = heatmap_pivot[sorted_columns]

    # Step 8: Plot heatmap
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Gross Margin (%)"),
        color_continuous_scale='Turbo',
        aspect="auto",
        text_auto=True
    )

    fig_heatmap.update_layout(
        title='Achieved Gross Margin (%) by Manager & Month',
        xaxis_side="top"
    )

    # Step 9: Show plot
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ------------------ REGION-WISE ANALYSIS ------------------
    st.subheader("🗺️ Region-wise Achieved Gross Margin (%)")
    country_data = filtered_df.groupby('Country')['Achieved Gross Margin (%)'].sum().reset_index()
    fig_choropleth = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Achieved Gross Margin (%)',
        color_continuous_scale='Plasma',
        title='Achieved Gross Margin (%) by Country',
        hover_name='Country'
    )
    fig_choropleth.update_geos(showframe=True, showcoastlines=True, projection_type="natural earth")
    fig_choropleth.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig_choropleth, use_container_width=True)

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
