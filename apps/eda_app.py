import plotly.express as px
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Boston Housing EDA Dashboard",
    layout="wide"
)

# Title
st.title("Boston Housing - Exploratory Data Analysis Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Boston.csv")
    # Drop unwanted index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

df = load_data()

# Show basic info
st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")


# Sidebar
st.sidebar.header("Controls")

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
numeric_cols.remove("medv")  # remove target from predictors

selected_var = st.sidebar.selectbox(
    "Select a variable to plot against MEDV:",
    numeric_cols
)


st.sidebar.subheader("Filters")
# Create range sliders for selected variable and MEDV
x_min, x_max = float(df[selected_var].min()), float(df[selected_var].max())
y_min, y_max = float(df["medv"].min()), float(df["medv"].max())

x_range = st.sidebar.slider(
    f"Range for {selected_var.upper()}",
    min_value=x_min,
    max_value=x_max,
    value=(x_min, x_max)
)

y_range = st.sidebar.slider(
    "Range for MEDV",
    min_value=y_min,
    max_value=y_max,
    value=(y_min, y_max)
)
filtered_df = df[
    (df[selected_var] >= x_range[0]) & (df[selected_var] <= x_range[1]) &
    (df["medv"] >= y_range[0]) & (df["medv"] <= y_range[1])
]



tab1, tab2, tab3, tab4 = st.tabs(
    ["Scatter Analysis", "Boxplots", "Distributions", "Summary & Correlation"]
)

with tab1:
    st.subheader(f"MEDV vs {selected_var.upper()}")


    filtered_df = df[
    (df[selected_var] >= x_range[0]) & (df[selected_var] <= x_range[1]) &
    (df["medv"] >= y_range[0]) & (df["medv"] <= y_range[1])
    ]
        
    fig = px.scatter(
    filtered_df,
    x=selected_var,
    y="medv",
    title=f"Median House Value vs {selected_var.upper()}",
    labels={selected_var: selected_var.upper(), "medv": "MEDV"},
    opacity=0.7
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Outlier Analysis (Boxplots)")

    box_var = st.selectbox(
        "Select variable for boxplot:",
        numeric_cols + ["medv"],
        key="boxplot_var"
    )

    fig_box = px.box(
        df,
        y=box_var,
        title=f"Boxplot of {box_var.upper()}",
        points="outliers"
    )

    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("Distribution of Variables")

    hist_var = st.selectbox(
        "Select variable for histogram:",
        numeric_cols + ["medv"],
        key="hist_var"
    )

    fig_hist = px.histogram(
        df,
        x=hist_var,
        nbins=30,
        title=f"Distribution of {hist_var.upper()}",
        marginal="box"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.subheader("Summary Statistics")

    st.write("Overall summary of numerical variables:")
    st.dataframe(df.describe())

    st.markdown("---")

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix Heatmap",
        color_continuous_scale="RdBu",
        origin="lower"
    )

    st.plotly_chart(fig_corr, use_container_width=True)






