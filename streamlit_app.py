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
    import pandas as pd
    import streamlit as st

    SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    SHEET_NAME = "Sheet1"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data(CSV_URL)

    # Convert 'Month-Year' to datetime
    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y")
    
    # Convert columns to appropriate types
    df["New Customer"] = df["New Customer"].fillna(0).astype(int)
    df["Committed Orders"] = pd.to_numeric(df["Committed Orders"], errors='coerce').fillna(0)
    df["Achieved Orders"] = pd.to_numeric(df["Achieved Orders"], errors='coerce').fillna(0)

    # Calculate conversion rate
    df["Conversion Rate (%)"] = df.apply(
        lambda row: (row["Achieved Orders"] / row["Committed Orders"] * 100) if row["Committed Orders"] else 0, axis=1
    )

    st.sidebar.header("🔎 Filters")

    # Sort Month-Year in chronological order for filtering
    month_options = sorted(df["Month-Year"].dropna().unique())

    # Convert back to 'Month-Year' string format for display purposes
    month_year = st.sidebar.multiselect("Select Month-Year(s):", options=month_options, format_func=lambda x: x.strftime('%b %Y'))

    # Filter the dataframe based on selected Month-Year
    if month_year:
        month_year_filtered = [pd.to_datetime(month, format='%b %Y') for month in month_year]
        filtered_df = df[df["Month-Year"].isin(month_year_filtered)].copy()
    else:
        filtered_df = df.copy()

    # Filter based on other sidebar options
    deal_managers = st.sidebar.multiselect("Select Deal Manager(s):", options=sorted(df["Deal Manager"].dropna().unique()))
    countries = st.sidebar.multiselect("Select Country(ies):", options=sorted(df["Country"].dropna().unique()))
    plants = st.sidebar.multiselect("Select Plant Type(s):", options=sorted(df["Plant Type"].dropna().unique()))
    customers = st.sidebar.multiselect("Select Customer(s):", options=sorted(df["Customer"].dropna().unique()))

    if deal_managers:
        filtered_df = filtered_df[filtered_df["Deal Manager"].isin(deal_managers)]
    if countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
    if plants:
        filtered_df = filtered_df[filtered_df["Plant Type"].isin(plants)]
    if customers:
        filtered_df = filtered_df[filtered_df["Customer"].isin(customers)]

    # Calculate metrics
    total_committed = filtered_df["Committed Orders"].sum()
    total_achieved = filtered_df["Achieved Orders"].sum()
    new_customers = filtered_df["New Customer"].sum()
    achieved_nonzero_df = filtered_df[filtered_df["Achieved Orders"] > 0]
    nonzero_achieved_count = len(achieved_nonzero_df)
    average_order_size = (
        achieved_nonzero_df["Achieved Orders"].sum() / nonzero_achieved_count
        if nonzero_achieved_count > 0 else 0
    )
    conversion_rate = (total_achieved / total_committed) * 100 if total_committed else 0


    # Display metrics
    #st.metric("Total Committed Orders", f"{total_committed:,.0f}")
    #st.metric("Total Achieved Orders", f"{total_achieved:,.0f}")
    #st.metric("New Customers", new_customers)
    #st.metric("Average Order Size", f"{average_order_size:,.2f}")
    #st.metric("Conversion Rate (%)", f"{conversion_rate:.2f}%")

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
        title="📊 Monthly Orders",
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
        color_continuous_scale='Plasma',
        custom_data=['Deal Manager', 'Plant Type', 'Customer', 'Achieved Orders'],
        title='Category-wise & Manager-wise Breakdown'
    )
    fig_treemap.update_traces(root_color="lightgrey")
    st.plotly_chart(fig_treemap, use_container_width=True)

    #selected_treemap = st.plotly_chart(fig_treemap, use_container_width=True)
    #st.info("🖱️ Click a Treemap section to drill down — feature for future interactivity.")

    # --- Heatmap with Drill-down ---
    st.subheader("🔥 Achieved Orders Heatmap (Manager × Month)")

    # Convert to datetime for proper chronological sorting
    filtered_df['Month_Year_Date'] = pd.to_datetime(filtered_df['Month-Year'], format='%b %Y')

    # Group data
    heatmap_data = filtered_df.groupby(['Deal Manager', 'Month_Year_Date'])['Achieved Orders'].sum().reset_index()

    # Pivot table
    heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month_Year_Date', values='Achieved Orders').fillna(0).round(2)

    # Sort columns chronologically
    heatmap_pivot = heatmap_pivot.sort_index(axis=1)

    # Add averages
    heatmap_pivot.loc['Average'] = heatmap_pivot.mean()
    heatmap_pivot['Average'] = heatmap_pivot.mean(axis=1)

    # Format column labels back to 'Mon YYYY'
    heatmap_pivot.columns = [col.strftime('%b %Y') if isinstance(col, pd.Timestamp) else col for col in heatmap_pivot.columns]

    # Plot
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Orders"),
        color_continuous_scale='Turbo',
        aspect="auto",
        text_auto=".2f"  # ✅ Correct way to show two decimal places
    )

    fig_heatmap.update_layout(
        #title='Achieved Orders by Manager & Month (with Averages)',
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
    import pandas as pd
    import streamlit as st

    SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    SHEET_NAME = "Revenue"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data(CSV_URL)

    # Convert 'Month-Year' to datetime format
    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y", errors='coerce')

    # Convert columns to appropriate types
    df["New Customer"] = df["New Customer"].fillna(0).astype(int)
    df["Committed Revenue"] = pd.to_numeric(df["Committed Revenue"], errors='coerce').fillna(0)
    df["Achieved Revenue"] = pd.to_numeric(df["Achieved Revenue"], errors='coerce').fillna(0)

    # Sidebar filters
    st.sidebar.header("🔎 Filters")

    # Sort Month-Year in chronological order for filtering
    month_options = sorted(df["Month-Year"].dropna().unique())

    # Convert back to 'Month-Year' string format for display purposes
    month_year = st.sidebar.multiselect("Select Month-Year(s):", options=month_options, format_func=lambda x: x.strftime('%b %Y'))

    # Filter the dataframe based on selected Month-Year
    if month_year:
        month_year_filtered = [pd.to_datetime(month, format='%b %Y') for month in month_year]
        filtered_df = df[df["Month-Year"].isin(month_year_filtered)]
    else:
        filtered_df = df.copy()

    # Filter based on other sidebar options
    deal_managers = st.sidebar.multiselect("Select Deal Manager(s):", options=sorted(df["Deal Manager"].dropna().unique()))
    countries = st.sidebar.multiselect("Select Country(ies):", options=sorted(df["Country"].dropna().unique()))
    plants = st.sidebar.multiselect("Select Plant Type(s):", options=sorted(df["Plant Type"].dropna().unique()))
    customers = st.sidebar.multiselect("Select Customer(s):", options=sorted(df["Customer"].dropna().unique()))

    if deal_managers:
        filtered_df = filtered_df[filtered_df["Deal Manager"].isin(deal_managers)]
    if countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
    if plants:
        filtered_df = filtered_df[filtered_df["Plant Type"].isin(plants)]
    if customers:
        filtered_df = filtered_df[filtered_df["Customer"].isin(customers)]

    # Calculate metrics
    total_committed = filtered_df["Committed Revenue"].sum()
    total_achieved = filtered_df["Achieved Revenue"].sum()
    new_customers = filtered_df["New Customer"].sum()
    average_revenue_size = (total_achieved / len(filtered_df)) if len(filtered_df) > 0 else 0
    conversion_rate = (total_achieved / total_committed) * 100 if total_committed else 0

    # Display metrics
    #st.metric("Total Committed Revenue", f"${total_committed:,.0f}")
    #st.metric("Total Achieved Revenue", f"${total_achieved:,.0f}")
    #st.metric("New Customers", new_customers)
    #st.metric("Average Revenue Size", f"${average_revenue_size:,.2f}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📌 Total Committed Revenue", f"${total_committed:,.0f}")
    col2.metric("✅ Total Achieved Revenue", f"${total_achieved:,.0f}")
    col3.metric("🎯 Conversion Rate", f"{conversion_rate:.2f}%")
    col4.metric("🆕 New Customers", f"{new_customers}")
    col5.metric("📦 Avg. Revenue Size", f"${average_revenue_size:,.0f}")

    # --- Monthly Revenue Comparison ---
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
        title="📊 Monthly Revenue",
        xaxis_title="Month-Year",
        yaxis_title="Revenue (USD)",
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        template="plotly_white",
        barmode='group',
        height=500
    )
    fig_revenue.update_yaxes(title_text="Revenue (USD)", secondary_y=False)
    #fig_revenue.update_yaxes(title_text="Conversion Rate (%)", secondary_y=True)

    st.plotly_chart(fig_revenue, use_container_width=True)

    # --- Treemap with Drill-down ---
    st.subheader("📘 Category-wise & Manager-wise Breakdown (Treemap)")

    # Drop rows with missing key hierarchy values
    treemap_df = filtered_df.dropna(subset=["Deal Manager", "Plant Type", "Customer", "Achieved Revenue"]).copy()

    # Ensure all path columns are strings
    treemap_df["Deal Manager"] = treemap_df["Deal Manager"].astype(str)
    treemap_df["Plant Type"] = treemap_df["Plant Type"].astype(str)
    treemap_df["Customer"] = treemap_df["Customer"].astype(str)

    # Also ensure Achieved Orders is numeric and non-negative
    treemap_df["Achieved Revenue"] = pd.to_numeric(treemap_df["Achieved Revenue"], errors='coerce').fillna(0)
    treemap_df = treemap_df[treemap_df["Achieved Revenue"] > 0]

    # Create the treemap
    fig_treemap = px.treemap(
        treemap_df,
        path=['Deal Manager', 'Plant Type', 'Customer'],
        values='Achieved Revenue',
        color='Achieved Revenue',
        color_continuous_scale='Plasma',
        title='Category-wise & Manager-wise Breakdown'
    )
    fig_treemap.update_traces(root_color="lightgrey")
    st.plotly_chart(fig_treemap, use_container_width=True)

    # --- Heatmap with Drill-down ---
    st.subheader("🔥 Achieved Revenue Heatmap (Manager × Month)")

    # Convert to datetime for sorting
    filtered_df['Month_Year_Date'] = pd.to_datetime(filtered_df['Month-Year'], format='%b %Y')

    # Group data
    heatmap_data = filtered_df.groupby(['Deal Manager', 'Month_Year_Date'])['Achieved Revenue'].sum().reset_index()

    # Pivot table
    heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month_Year_Date', values='Achieved Revenue').fillna(0).round(2)

    # Sort columns chronologically
    heatmap_pivot = heatmap_pivot.sort_index(axis=1)

    # Add averages
    heatmap_pivot.loc['Average'] = heatmap_pivot.mean()
    heatmap_pivot['Average'] = heatmap_pivot.mean(axis=1)

    # Format column labels back to 'Mon YYYY'
    heatmap_pivot.columns = [col.strftime('%b %Y') if isinstance(col, pd.Timestamp) else col for col in heatmap_pivot.columns]

    # Plot heatmap
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Revenue"),
        color_continuous_scale='Turbo',
        aspect="auto",
        text_auto=".2f"
    )
    fig_heatmap.update_layout(
        #title='Achieved Revenue by Manager & Month (with Averages)',
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
    SHEET_NAME = "GrossM"
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
    #col1, col2, col3 = st.columns(3)

    #col1.metric("Achieved Gross Margin (USD)", f"${Achieved Gross Margin (USD):.2f}")
    #col2.metric("Achieved Gross Margin (%)", f"{Achieved Gross Margin (%):.2f}%")
    #col3.metric("Margin Realization (%)", f"{Margin Realization (%):.2f}%")


    # ------------------ SIDEBAR FILTERS ------------------
    # Convert and clean data
    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y", errors='coerce')
    df = df.dropna(subset=["Month-Year"])  # Ensure datetime format only

    df["New Customer"] = df["New Customer"].fillna(0).astype(int)
    df["Committed Revenue"] = pd.to_numeric(df["Committed Revenue"], errors='coerce').fillna(0)
    df["Achieved Revenue"] = pd.to_numeric(df["Achieved Revenue"], errors='coerce').fillna(0)

    # Sidebar Filters
    st.sidebar.header("🔎 Filters")

    month_year_options = sorted(df["Month-Year"].unique())
    month_year = st.sidebar.multiselect(
        "Select Month-Year(s):",
        options=month_year_options,
        format_func=lambda x: x.strftime('%b %Y')  # Display nicely
    )

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

    # Calculations
    total_committed = filtered_df["Committed Revenue"].sum()
    total_achieved = filtered_df["Achieved Revenue"].sum()
    new_customers = filtered_df["New Customer"].sum()
    average_revenue_size = (total_achieved / len(filtered_df)) if len(filtered_df) > 0 else 0


    # ------------------ VISUALIZATION ------------------
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

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

    #monthly["Margin Realization (%)"] = np.where(
        #(monthly["Committed Gross Margin (USD)"] != 0) & (monthly["Achieved Gross Margin (USD)"] != 0),
        #(monthly["Achieved Gross Margin (USD)"] / monthly["Committed Gross Margin (USD)"]) * 100,
        #0
    #)

    # Plot
    fig1 = go.Figure()

    # Committed Gross Margin (%)
    fig1.add_trace(go.Bar(
        x=monthly["Month-Year"],
        y=monthly["Committed Gross Margin (%)"],
        name="Committed GM (%)",
        marker_color='lightblue',
        yaxis="y",
        hovertemplate="Month: %{x}<br>Committed GM: %{y:.1f}%"
    ))

    # Achieved Gross Margin (%)
    fig1.add_trace(go.Bar(
        x=monthly["Month-Year"],
        y=monthly["Achieved Gross Margin (%)"],
        name="Achieved GM (%)",
        marker_color='green',
        yaxis="y",
        hovertemplate="Month: %{x}<br>Achieved GM: %{y:.1f}%"
    ))

    # Achieved Gross Margin in USD (Line Plot, right axis)
    fig1.add_trace(go.Scatter(
        x=monthly["Month-Year"],
        y=monthly["Achieved Gross Margin (USD)"],
        name="Achieved GM (USD)",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="orange"),
        hovertemplate="Month: %{x}<br>Achieved GM (USD): $%{y:,.0f}"
    ))

    # Layout
    fig1.update_layout(
        title="Monthly Gross Margin (%) and Gross Margin (USD)",
        xaxis_title="Month-Year",
        yaxis=dict(
            title="Gross Margin (%)",
            range=[0, max(monthly[["Committed Gross Margin (%)", "Achieved Gross Margin (%)"]].max()) + 10],
            showgrid=False
        ),
        yaxis2=dict(
            title="Achieved GM (USD)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=500
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
        color_continuous_scale='Plasma',
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
    agg_data = (
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

    # Step 4: Compute Achieved Gross Margin (%)
    agg_data['Achieved Gross Margin (%)'] = (
        (agg_data['Achieved Revenue'] - (
            agg_data['Achieved COGS'] +
            agg_data['Achieved Logistics'] +
            agg_data['Achieved P&F'] +
            agg_data['Achieved Associate Payment']
        )) / agg_data['Achieved Revenue']
    ) * 100

    # Step 5: Format 'Month-Year' for display
    agg_data['Month-Year'] = agg_data['MonthYearSort'].dt.strftime('%b %Y')

    # Step 6: Pivot data for Achieved Revenue & Cost Components for correct summation
    pivot_components = agg_data.pivot(index='Deal Manager', columns='Month-Year', values=[
        'Achieved Revenue', 'Achieved COGS', 'Achieved Logistics', 'Achieved P&F', 'Achieved Associate Payment']
    )

    # Step 7: Recalculate row-wise average gross margin (%) per Deal Manager
    row_avg = (
        (pivot_components['Achieved Revenue'].sum(axis=1) - (
            pivot_components['Achieved COGS'].sum(axis=1) +
            pivot_components['Achieved Logistics'].sum(axis=1) +
            pivot_components['Achieved P&F'].sum(axis=1) +
            pivot_components['Achieved Associate Payment'].sum(axis=1)
        )) / pivot_components['Achieved Revenue'].sum(axis=1)
    ) * 100
    row_avg = row_avg.round(2)

    # Step 8: Recalculate column-wise average gross margin (%) per Month-Year
    col_avg = (
        (pivot_components['Achieved Revenue'].sum(axis=0) - (
            pivot_components['Achieved COGS'].sum(axis=0) +
            pivot_components['Achieved Logistics'].sum(axis=0) +
            pivot_components['Achieved P&F'].sum(axis=0) +
            pivot_components['Achieved Associate Payment'].sum(axis=0)
        )) / pivot_components['Achieved Revenue'].sum(axis=0)
    ) * 100
    col_avg = col_avg.round(2)

    # Step 9: Create final heatmap data pivot for Achieved Gross Margin (%)
    heatmap_pivot = agg_data.pivot(index='Deal Manager', columns='Month-Year', values='Achieved Gross Margin (%)').fillna(0).round(2)

    # Step 10: Append 'Average' row and column
    heatmap_pivot['Average'] = row_avg  # row-wise average
    col_avg['Average'] = (  # dummy value for bottom-right corner
        (row_avg * pivot_components['Achieved Revenue'].sum(axis=1)).sum()
        /
        pivot_components['Achieved Revenue'].sum().sum()
    )  # full-table weighted average
    heatmap_pivot.loc['Average'] = col_avg.round(2)

    # Step 11: Sort columns chronologically (including 'Average' last)
    sorted_columns = sorted(
        [col for col in heatmap_pivot.columns if col != 'Average'],
        key=lambda x: pd.to_datetime(x, format='%b %Y')
    )
    sorted_columns.append('Average')
    heatmap_pivot = heatmap_pivot[sorted_columns]

    # Step 12: Plot heatmap
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Gross Margin (%)"),
        color_continuous_scale='Turbo',
        aspect="auto",
        text_auto=".2f"
    )

    fig_heatmap.update_layout(
        xaxis_side="top"
    )

    # Step 13: Show plot
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ------------------ REGION-WISE ANALYSIS ------------------
    import streamlit as st
    import plotly.express as px
    import pandas as pd

    st.subheader("🗺️ Region-wise Achieved Gross Margin (%)")

    # Step 1: Calculate Achieved Gross Margin per row
    filtered_df['Achieved Gross Margin (%)'] = (
        (filtered_df['Achieved Revenue'] - (
            filtered_df['Achieved COGS'] +
            filtered_df['Achieved Logistics'] +
            filtered_df['Achieved P&F'] +
            filtered_df['Achieved Associate Payment']
        )) / filtered_df['Achieved Revenue']
    ) * 100

    # Step 2: Compute weighted average Gross Margin per country
    # Weighted by Achieved Revenue
    country_data = filtered_df.groupby('Country').apply(
        lambda x: ((
            x['Achieved Revenue'] - (
                x['Achieved COGS'] +
                x['Achieved Logistics'] +
                x['Achieved P&F'] +
                x['Achieved Associate Payment']
            )
        ).sum() / x['Achieved Revenue'].sum()) * 100
    ).reset_index(name='Achieved Gross Margin (%)')

    # Step 3: Create choropleth map
    fig_choropleth = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Achieved Gross Margin (%)',
        color_continuous_scale='Plasma',
        hover_name='Country'
    )

    # Step 4: Update map visuals
    fig_choropleth.update_geos(
        showframe=True,
        showcoastlines=True,
        projection_type="natural earth"
    )

    fig_choropleth.update_layout(
        title_text='Achieved Gross Margin (%) by Country',
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )

    # Step 5: Render in Streamlit
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
