import streamlit as st
import pandas as pd
import plotly.express as px

# Load preprocessed data
@st.cache_data
def load_data():
    return pd.read_excel("final_output.xlsx")

df = load_data()

# Convert relevant columns to strings
columns_to_string = ['Financial Year', 'Customer Group', 'Country', 'Region', 'Sales Manager', 'Business Unit']
for col in columns_to_string:
    df[col] = df[col].astype(str)

# Sidebar filters
st.sidebar.title("Filters")
filters = {}
for col in columns_to_string:
    options = df[col].unique().tolist()
    selection = st.sidebar.multiselect(f"Select {col}", options, default=options)
    filters[col] = selection

# Filter the dataframe
filtered_df = df.copy()
for col, selected_vals in filters.items():
    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

# Toggle for Grouped or Stacked Bar
chart_mode = st.radio("Chart Mode", ['Grouped', 'Stacked'], horizontal=True)

# Gross Margin Toggle
gm_view = st.radio("Gross Margin View", ['By Value', 'By %'], horizontal=True)

# Define chart drawing functions
def draw_bar_chart(df, y1, y2, title, yaxis_title):
    df = df.groupby('Financial Year')[[y1, y2]].sum().reset_index()
    df_melted = df.melt(id_vars='Financial Year', value_vars=[y1, y2], var_name='Metric', value_name='Value')
    barmode = 'group' if chart_mode == 'Grouped' else 'stack'
    fig = px.bar(df_melted, x='Financial Year', y='Value', color='Metric', barmode=barmode, title=title)
    fig.update_layout(yaxis_title=yaxis_title)
    return fig

# Orders Comparison
st.subheader("📦 Orders Comparison")
st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Orders', 'Achieved Orders', "Orders Comparison", "Orders"))

# Revenue Comparison
st.subheader("💰 Revenue Comparison")
st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Revenue', 'Achieved Revenue', "Revenue Comparison", "Revenue"))

# Gross Margin Comparison
st.subheader("📈 Gross Margin Comparison")
if gm_view == 'By Value':
    st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Gross Margin Value', 'Achieved Gross Margin Value', "Gross Margin (Value)", "Gross Margin Value"))
else:
    st.plotly_chart(draw_bar_chart(filtered_df, 'Committed Gross Margin %', 'Achieved Gross Margin %', "Gross Margin (%)", "Gross Margin %"))

# Delta Summary Table
st.subheader("📊 Delta Summary Table (Achieved – Committed)")
summary = filtered_df.groupby('Financial Year').agg({
    'Achieved Orders': 'sum',
    'Committed Orders': 'sum',
    'Achieved Revenue': 'sum',
    'Committed Revenue': 'sum',
    'Achieved Gross Margin Value': 'sum',
    'Committed Gross Margin Value': 'sum',
    'Achieved Gross Margin %': 'mean',
    'Committed Gross Margin %': 'mean'
}).reset_index()

summary['Delta Orders'] = summary['Achieved Orders'] - summary['Committed Orders']
summary['Delta Revenue'] = summary['Achieved Revenue'] - summary['Committed Revenue']
summary['Delta GM Value'] = summary['Achieved Gross Margin Value'] - summary['Committed Gross Margin Value']
summary['Delta GM %'] = summary['Achieved Gross Margin %'] - summary['Committed Gross Margin %']

delta_table = summary[['Financial Year', 'Delta Orders', 'Delta Revenue', 'Delta GM Value', 'Delta GM %']]
st.dataframe(delta_table.style.format({
    'Delta Orders': '{:,.0f}',
    'Delta Revenue': '{:,.0f}',
    'Delta GM Value': '{:,.0f}',
    'Delta GM %': '{:.2f}%'
}))
