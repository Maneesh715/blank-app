import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import BytesIO
from urllib.parse import quote

st.set_page_config(page_title="Worldref Dashboard", layout="wide")

st.sidebar.title("📁 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["📊 Orders", "📊 Revenue", "📊 Gross Margin"],
    index=0  # Default to first dashboard
)

if page == "📊 Orders":
    import pandas as pd
    import streamlit as st

    SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    SHEET_NAME = "Orders"
    CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    @st.cache_data
    def load_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df

    df = load_data(CSV_URL)

    # Convert 'Month-Year' to datetime
    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format='mixed', errors='coerce')
    
    # Convert columns to appropriate types
    df["New Customer"] = df["New Customer"].fillna(0).astype(int)
    df["Committed Order Booking"] = pd.to_numeric(df["Committed Order Booking"], errors='coerce').fillna(0)
    df["Achieved Order Booking"] = pd.to_numeric(df["Achieved Order Booking"], errors='coerce').fillna(0)

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
    total_committed = filtered_df["Committed Order Booking"].sum()
    total_achieved = filtered_df["Achieved Order Booking"].sum()
    new_customers = filtered_df["New Customer"].sum()
    achieved_nonzero_df = filtered_df[filtered_df["Achieved Order Booking"] > 0]
    nonzero_achieved_count = len(achieved_nonzero_df)
    average_order_size = (
        achieved_nonzero_df["Achieved Order Booking"].sum() / nonzero_achieved_count
        if nonzero_achieved_count > 0 else 0
    )

    col1, col2, col4, col5 = st.columns(4)
    col1.metric("📌 Committed Order Booking", f"$ {total_committed / 1_000_000:.2f} Mn")
    col2.metric("✅ Achieved Order Booking", f"${total_achieved:,.0f}")
    col4.metric("🆕 New Customers", f"{new_customers}")
    col5.metric("📦 Avg. Order Size", f"${average_order_size:,.0f}")

    # --- Monthly Orders Comparison ---
    monthly_summary = (
        filtered_df.groupby(filtered_df["Month-Year"].dt.to_period("M"))[["Committed Order Booking", "Achieved Order Booking"]]
        .sum()
        .reset_index()
    )
    monthly_summary["Month-Year"] = monthly_summary["Month-Year"].dt.strftime("%b'%y")

    fig_orders = make_subplots(specs=[[{"secondary_y": True}]])
    fig_orders.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Committed Order Booking"],
                                name="Committed Order Booking", marker_color="#66c2a5", text=monthly_summary["Committed Order Booking"], textposition='outside'),
                         secondary_y=False)
    fig_orders.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Achieved Order Booking"],
                                name="Achieved Order Booking", marker_color="#1d3557", text=monthly_summary["Achieved Order Booking"], textposition='outside'),
                         secondary_y=False)

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

    st.plotly_chart(fig_orders, use_container_width=True)

    # --- Hero Customers Section ---
    st.subheader("🏅 Hero Customers (Quarter-wise)")

    HERO_SHEET_ID = "1VGd-4Ycj8mz8ZvDV2chLt4bG8DMjQ64fSLADkmXLsPo"
    HERO_SHEET_NAME = quote("Hero Customers")
    HERO_CSV_URL = f"https://docs.google.com/spreadsheets/d/{HERO_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={HERO_SHEET_NAME}"

    @st.cache_data
    def load_hero_data(url):
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='mixed', errors='coerce')
        return df

    try:
        hero_df = load_hero_data(HERO_CSV_URL)

        # Assign quarter label
        hero_df['Quarter'] = hero_df['Month-Year'].dt.to_period('Q').astype(str)

        # Aggregate per customer per quarter
        agg_hero = (
            hero_df.groupby(['Quarter', 'Customer'])
            .agg({'Enquiry Count': 'sum', 'Orders Count': 'sum'})
            .reset_index()
        )
        agg_hero['Win Rate'] = agg_hero['Orders Count'] / agg_hero['Enquiry Count']

        # Apply Hero condition
        hero_filter = (agg_hero['Enquiry Count'] >= 15) & (agg_hero['Win Rate'] >= 0.15)
        heroes = agg_hero[hero_filter]

        # Format output
        quarter_heroes = heroes.groupby('Quarter')['Customer'].apply(lambda x: ', '.join(sorted(x.unique()))).reset_index()
        for _, row in quarter_heroes.iterrows():
            st.markdown(f"**{row['Quarter']}**: {row['Customer']}")

    except Exception as e:
        st.warning("Could not load Hero Customers data. Please check the Google Sheet link.")
        st.error(str(e))

    # --- Treemap with Drill-down ---
    st.subheader("📘 Category-wise & Manager-wise Breakdown (Treemap)")

    # Remove rows with missing hierarchy levels
    filtered_df = filtered_df.dropna(subset=['Deal Manager', 'Plant Type', 'Customer'])

    fig_treemap = px.treemap(
        filtered_df,
        path=['Deal Manager', 'Plant Type', 'Customer'],
        values='Achieved Order Booking',
        color='Achieved Order Booking',
        color_continuous_scale='Plasma',
        custom_data=['Deal Manager', 'Plant Type', 'Customer', 'Achieved Order Booking'],
        title='Category-wise & Manager-wise Breakdown'
    )
    fig_treemap.update_traces(root_color="lightgrey")

    st.plotly_chart(fig_treemap, use_container_width=True)

    # --- Heatmap with Drill-down ---
    st.subheader("🔥 Achieved Order Booking Heatmap (Manager × Month)")

    # Convert to datetime for proper chronological sorting
    filtered_df['Month_Year_Date'] = pd.to_datetime(filtered_df['Month-Year'], format='%b %Y')

    # Group data
    heatmap_data = filtered_df.groupby(['Deal Manager', 'Month_Year_Date'])['Achieved Order Booking'].sum().reset_index()

    # Pivot table
    heatmap_pivot = heatmap_data.pivot(index='Deal Manager', columns='Month_Year_Date', values='Achieved Order Booking').fillna(0).round(2)

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
        labels=dict(x="Month-Year", y="Deal Manager", color="Achieved Order Booking"),
        color_continuous_scale='Turbo',
        aspect="auto",
        text_auto=".2f"
    )

    fig_heatmap.update_layout(
        xaxis_side="top"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- Map ---
    st.subheader("🗺️ Region-wise Achieved Order Booking")
    country_data = filtered_df.groupby('Country')['Achieved Order Booking'].sum().reset_index()
    fig_choropleth = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Achieved Order Booking',
        color_continuous_scale='Plasma',
        title='Achieved Order Booking by Country',
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

elif page == "📊 Revenue":
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
    #df["Committed Revenue"] = pd.to_numeric(df["Committed Revenue"], errors='coerce').fillna(0)
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
    #total_committed = filtered_df["Committed Revenue"].sum()
    total_achieved = filtered_df["Achieved Revenue"].sum()
    new_customers = filtered_df["New Customer"].sum()
    achieved_nonzero_df = filtered_df[filtered_df["Achieved Revenue"] > 0]
    nonzero_achieved_count = len(achieved_nonzero_df)
    average_revenue_size = (
        achieved_nonzero_df["Achieved Revenue"].sum() / nonzero_achieved_count
        if nonzero_achieved_count > 0 else 0
    )
    #conversion_rate = (total_achieved / total_committed) * 100 if total_committed else 0

    # Display metrics
    #st.metric("Total Committed Revenue", f"${total_committed:,.0f}")
    #st.metric("Total Achieved Revenue", f"${total_achieved:,.0f}")
    #st.metric("New Customers", new_customers)
    #st.metric("Average Revenue Size", f"${average_revenue_size:,.2f}")

    col2, col4, col5 = st.columns(3)
    #col1.metric("📌 Total Committed Revenue", f"${total_committed:,.0f}")
    col2.metric("✅ Achieved Revenue", f"${total_achieved:,.0f}")
    #col3.metric("🎯 Conversion Rate", f"{conversion_rate:.2f}%")
    col4.metric("🆕 New Customers", f"{new_customers}")
    col5.metric("📦 Avg. Revenue Size", f"${average_revenue_size:,.0f}")

    # --- Monthly Revenue Comparison ---
    monthly_summary = (
        filtered_df.groupby(filtered_df["Month-Year"].dt.to_period("M"))[["Achieved Revenue"]]
        .sum()
        .reset_index()
    )
    monthly_summary["Month-Year"] = monthly_summary["Month-Year"].dt.strftime("%b'%y")
    #monthly_summary["Conversion Rate (%)"] = monthly_summary.apply(
        #lambda row: (row["Achieved Revenue"] / row["Committed Revenue"] * 100) if row["Committed Revenue"] else 0,
        #axis=1
    #)

    fig_revenue = make_subplots(specs=[[{"secondary_y": True}]])
    #fig_revenue.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Committed Revenue"],
                                #name="Committed Revenue", marker_color="#66c2a5", text=monthly_summary["Committed Revenue"], textposition='outside'),
                         #secondary_y=False)
    fig_revenue.add_trace(go.Bar(x=monthly_summary["Month-Year"], y=monthly_summary["Achieved Revenue"],
                                name="Achieved Revenue", marker_color="#1d3557", text=monthly_summary["Achieved Revenue"], textposition='outside'),
                         secondary_y=False)
    #fig_revenue.add_trace(go.Scatter(x=monthly_summary["Month-Year"], y=monthly_summary["Conversion Rate (%)"],
                                    #name="Conversion Rate (%)", mode='lines+markers', line=dict(color="#e76f51", width=3), marker=dict(size=6)),
                         #secondary_y=True)

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
    #st.set_page_config(page_title="Gross Margin", layout="wide")
    #st.title("📊 Gross Margin")

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
        "Booked COGS", "Realized COGS", "Booked Logistics", "Realized Logistics",
        "Booked P&F", "Realized P&F", "Booked Associate Payment", "Realized Associate Payment",
        "Booked Revenue", "Realized Revenue"
    ]

    for col in value_cols_inr:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col + " (USD)"] = df[col] / usd_conversion

    df["Booked Gross Margin (USD)"] = df["Booked Revenue (USD)"] - (
        df["Booked COGS (USD)"] + df["Booked Logistics (USD)"] + df["Booked P&F (USD)"] + df["Booked Associate Payment (USD)"])

    df["Realized Gross Margin (USD)"] = df["Realized Revenue (USD)"] - (
        df["Realized COGS (USD)"] + df["Realized Logistics (USD)"] + df["Realized P&F (USD)"] + df["Realized Associate Payment (USD)"])

    df["Booked Gross Margin (%)"] = np.where(df["Booked Revenue (USD)"] > 0,
        (df["Booked Gross Margin (USD)"] / df["Booked Revenue (USD)"]) * 100, np.nan)
    df["Realized Gross Margin (%)"] = np.where(df["Realized Revenue (USD)"] > 0,
        (df["Realized Gross Margin (USD)"] / df["Realized Revenue (USD)"]) * 100, np.nan)
    #df["Margin Realization (%)"] = np.where(df["Committed Gross Margin (USD)"] > 0,
        #(df["Achieved Gross Margin (USD)"] / df["Committed Gross Margin (USD)"]) * 100, np.nan)

    # Display KPI cards in 3 columns
    #col1, col4 = st.columns(2)

    #col1.metric("Achieved Gross Margin (USD)", f"${Achieved Gross Margin (USD):.2f}")
    #col4.metric("Achieved Gross Margin (%)", f"{Achieved Gross Margin (%):.2f}%")
    #col3.metric("Margin Realization (%)", f"{Margin Realization (%):.2f}%")


    # ------------------ SIDEBAR FILTERS ------------------
    # Convert and clean data
    df["Month-Year"] = pd.to_datetime(df["Month-Year"], format="%b %Y", errors='coerce')
    df = df.dropna(subset=["Month-Year"])  # Ensure datetime format only

    df["New Customer"] = df["New Customer"].fillna(0).astype(int)
    df["Booked Revenue"] = pd.to_numeric(df["Booked Revenue"], errors='coerce').fillna(0)
    df["Realized Revenue"] = pd.to_numeric(df["Realized Revenue"], errors='coerce').fillna(0)

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
    total_committed = filtered_df["Booked Revenue"].sum()
    total_achieved = filtered_df["Realized Revenue"].sum()
    new_customers = filtered_df["New Customer"].sum()
    average_revenue_size = (total_achieved / len(filtered_df)) if len(filtered_df) > 0 else 0


    # ------------------ VISUALIZATION ------------------
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go

    st.header("📅 Monthly Gross Margin")

    # Add sorting column
    filtered_df["MonthYearSort"] = pd.to_datetime(filtered_df["Month-Year"], format="%b %Y", errors='coerce')

    # Filter out rows with zero Achieved Revenue (USD)
    filtered_df = filtered_df[filtered_df["Realized Revenue (USD)"] != 0]

    # Aggregate
    monthly = (
        filtered_df.groupby("MonthYearSort")
        .agg({
            "Booked Gross Margin (USD)": "sum",
            "Realized Gross Margin (USD)": "sum",
            "Booked Revenue (USD)": "sum",
            "Realized Revenue (USD)": "sum"
        })
        .reset_index()
    )

    monthly["Month-Year"] = monthly["MonthYearSort"].dt.strftime("%b %Y")

    # Calculate %
    monthly["Booked Gross Margin (%)"] = np.where(
        monthly["Booked Revenue (USD)"] != 0,
        (monthly["Booked Gross Margin (USD)"] / monthly["Booked Revenue (USD)"]) * 100,
        0
    )

    monthly["Realized Gross Margin (%)"] = np.where(
        monthly["Realized Revenue (USD)"] != 0,
        (monthly["Realized Gross Margin (USD)"] / monthly["Realized Revenue (USD)"]) * 100,
        0
    )

    #monthly["Margin Realization (%)"] = np.where(
        #(monthly["Committed Gross Margin (USD)"] != 0) & (monthly["Achieved Gross Margin (USD)"] != 0),
        #(monthly["Achieved Gross Margin (USD)"] / monthly["Committed Gross Margin (USD)"]) * 100,
        #0
    #)

    # Plot
    fig1 = go.Figure()

    # Booked Gross Margin (%)
    fig1.add_trace(go.Bar(
        x=monthly["Month-Year"],
        y=monthly["Booked Gross Margin (%)"],
        name="Booked GM (%)",
        marker_color='lightcoral',
        yaxis="y",
        hovertemplate="Month: %{x}<br>Committed GM: %{y:.1f}%",
        text=[f"{val:.2f}%" for val in monthly["Booked Gross Margin (%)"]],
        textposition="outside",  # puts text on top of the bar
        texttemplate="%{text}"
    ))

    # Realized Gross Margin (%)
    fig1.add_trace(go.Bar(
        x=monthly["Month-Year"],
        y=monthly["Realized Gross Margin (%)"],
        name="Realized GM (%)",
        marker_color='green',
        yaxis="y",
        hovertemplate="Month: %{x}<br>Realized GM: %{y:.1f}%",
        text=[f"{val:.2f}%" for val in monthly["Realized Gross Margin (%)"]],
        textposition="outside",  # puts text on top of the bar
        texttemplate="%{text}"
    ))

    # Realized Gross Margin in USD (Line Plot, right axis)
    fig1.add_trace(go.Scatter(
        x=monthly["Month-Year"],
        y=monthly["Realized Gross Margin (USD)"],
        name="Realized GM (USD)",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="orange"),
        hovertemplate="Month: %{x}<br>Realized GM (USD): $%{y:,.0f}"
    ))

    # Layout
    fig1.update_layout(
        #title="Monthly Gross Margin (%) and Gross Margin (USD)",
        xaxis_title="Month-Year",
        yaxis=dict(
            title="Gross Margin (%)",
            range=[0, max(monthly[["Booked Gross Margin (%)", "Realized Gross Margin (%)"]].max()) + 10],
            showgrid=False
        ),
        yaxis2=dict(
            title="Realized GM (USD)",
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
    if 'Realized Gross Margin (%)' not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['Realized Revenue'] = filtered_df['Realized Revenue'].replace(0, np.nan)
        filtered_df['Realized Gross Margin (%)'] = (
            (filtered_df['Realized Revenue'] - (
                filtered_df['Realized COGS'] + 
                filtered_df['Realized Logistics'] + 
                filtered_df['Realized P&F'] + 
                filtered_df['Realized Associate Payment']
            )) / filtered_df['Realized Revenue']
        ) * 100

    # ✅ Step 2: Fill missing values in metric and dimension columns
    filtered_df['Realized Gross Margin (%)'] = filtered_df['Realized Gross Margin (%)'].fillna(0)
    filtered_df[['Deal Manager', 'Plant Type', 'Customer']] = filtered_df[['Deal Manager', 'Plant Type', 'Customer']].fillna('Unknown')

    # ✅ Step 3: Replace zeros with a small positive value to make them visible in the treemap
    filtered_df['Realized Gross Margin Display'] = filtered_df['Realized Gross Margin (%)'].apply(lambda x: x if x > 0 else 0.01)

    # ✅ Step 4: Create the treemap
    fig_treemap = px.treemap(
        filtered_df,
        path=['Deal Manager', 'Plant Type', 'Customer'],
        values='Realized Gross Margin Display',
        color='Realized Gross Margin (%)',
        color_continuous_scale='Plasma',
        custom_data=['Deal Manager', 'Plant Type', 'Customer', 'Realized Gross Margin (%)'],
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
    st.subheader("🔥 Realized Gross Margin (%) Heatmap (Manager × Month)")

    # Step 1: Filter rows where Achieved Revenue is not zero
    filtered_df = df[df['Realized Revenue'] != 0].copy()

    # Step 2: Create 'MonthYearSort' from 'Month-Year' for sorting
    filtered_df['MonthYearSort'] = pd.to_datetime(filtered_df['Month-Year'], format='%b %Y')

    # Step 3: Aggregate raw financials by Deal Manager and MonthYearSort
    agg_data = (
        filtered_df.groupby(['Deal Manager', 'MonthYearSort'])
        .agg({
            'Realized Revenue': 'sum',
            'Realized COGS': 'sum',
            'Realized Logistics': 'sum',
            'Realized P&F': 'sum',
            'Realized Associate Payment': 'sum'
        })
        .reset_index()
    )

    # Step 4: Compute Achieved Gross Margin (%)
    agg_data['Realized Gross Margin (%)'] = (
        (agg_data['Realized Revenue'] - (
            agg_data['Realized COGS'] +
            agg_data['Realized Logistics'] +
            agg_data['Realized P&F'] +
            agg_data['Realized Associate Payment']
        )) / agg_data['Realized Revenue']
    ) * 100

    # Step 5: Format 'Month-Year' for display
    agg_data['Month-Year'] = agg_data['MonthYearSort'].dt.strftime('%b %Y')

    # Step 6: Pivot data for Achieved Revenue & Cost Components for correct summation
    pivot_components = agg_data.pivot(index='Deal Manager', columns='Month-Year', values=[
        'Realized Revenue', 'Realized COGS', 'Realized Logistics', 'Realized P&F', 'Realized Associate Payment']
    )

    # Step 7: Recalculate row-wise average gross margin (%) per Deal Manager
    row_avg = (
        (pivot_components['Realized Revenue'].sum(axis=1) - (
            pivot_components['Realized COGS'].sum(axis=1) +
            pivot_components['Realized Logistics'].sum(axis=1) +
            pivot_components['Realized P&F'].sum(axis=1) +
            pivot_components['Realized Associate Payment'].sum(axis=1)
        )) / pivot_components['Realized Revenue'].sum(axis=1)
    ) * 100
    row_avg = row_avg.round(2)

    # Step 8: Recalculate column-wise average gross margin (%) per Month-Year
    col_avg = (
        (pivot_components['Realized Revenue'].sum(axis=0) - (
            pivot_components['Realized COGS'].sum(axis=0) +
            pivot_components['Realized Logistics'].sum(axis=0) +
            pivot_components['Realized P&F'].sum(axis=0) +
            pivot_components['Realized Associate Payment'].sum(axis=0)
        )) / pivot_components['Realized Revenue'].sum(axis=0)
    ) * 100
    col_avg = col_avg.round(2)

    # Step 9: Create final heatmap data pivot for Achieved Gross Margin (%)
    heatmap_pivot = agg_data.pivot(index='Deal Manager', columns='Month-Year', values='Realized Gross Margin (%)').fillna(0).round(2)

    # Step 10: Append 'Average' row and column
    heatmap_pivot['Average'] = row_avg  # row-wise average
    col_avg['Average'] = (  # dummy value for bottom-right corner
        (row_avg * pivot_components['Realized Revenue'].sum(axis=1)).sum()
        /
        pivot_components['Realized Revenue'].sum().sum()
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
        labels=dict(x="Month-Year", y="Deal Manager", color="Realized Gross Margin (%)"),
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

    st.subheader("🗺️ Region-wise Realized Gross Margin (%)")

    # Step 1: Calculate Realized Gross Margin per row
    filtered_df['Realized Gross Margin (%)'] = (
        (filtered_df['Realized Revenue'] - (
            filtered_df['Realized COGS'] +
            filtered_df['Realized Logistics'] +
            filtered_df['Realized P&F'] +
            filtered_df['Realized Associate Payment']
        )) / filtered_df['Realized Revenue']
    ) * 100

    # Step 2: Compute weighted average Gross Margin per country
    # Weighted by Achieved Revenue
    country_data = filtered_df.groupby('Country').agg({
        'Realized Revenue': 'sum',
        'Realized COGS': 'sum',
        'Realized Logistics': 'sum',
        'Realized P&F': 'sum',
        'Realized Associate Payment': 'sum'
    }).reset_index()

    country_data['Realized Gross Margin (%)'] = (
        (country_data['Realized Revenue'] - (
            country_data['Realized COGS'] +
            country_data['Realized Logistics'] +
            country_data['Realized P&F'] +
            country_data['Realized Associate Payment']
        )) / country_data['Realized Revenue']
    ) * 100

    # Step 3: Create choropleth map
    fig_choropleth = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Realized Gross Margin (%)',
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
        title_text='Realized Gross Margin (%) by Country',
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
