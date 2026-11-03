
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RetailPulse AI",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "feature_engineered_sales.csv"
RESULTS_PATH = BASE_DIR / "outputs" / "model_comparison.csv"
IMPORTANCE_PATH = BASE_DIR / "outputs" / "feature_importance.csv"


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data


@st.cache_data
def load_model_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(IMPORTANCE_PATH)


try:
    df = load_data()
    results = load_model_results()
    feature_importance = load_feature_importance()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.title("📊 RetailPulse AI")
st.subheader("Intelligent Retail Sales & Demand Forecasting System")

st.markdown(
    "Analyze historical retail sales, understand demand patterns "
    "and explore machine learning model performance."
)

st.divider()


# --------------------------------------------------
# 5. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("Dashboard Filters")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

selected_stores = st.sidebar.multiselect(
    "Select Stores",
    options=sorted(df["store"].unique()),
    default=sorted(df["store"].unique())
)

selected_items = st.sidebar.multiselect(
    "Select Items",
    options=sorted(df["item"].unique()),
    default=sorted(df["item"].unique())
)


# --------------------------------------------------
# 6. APPLY FILTERS
# --------------------------------------------------

if len(selected_dates) != 2:
    st.warning("Please select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates

filtered_df = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date) &
    (df["store"].isin(selected_stores)) &
    (df["item"].isin(selected_items))
]

if filtered_df.empty:
    st.warning("No sales records found for the selected filters.")
    st.stop()


# --------------------------------------------------
# 7. SALES OVERVIEW
# --------------------------------------------------

st.header("Sales Overview")

total_sales = filtered_df["sales"].sum()
average_sales = filtered_df["sales"].mean()
total_stores = filtered_df["store"].nunique()
total_items = filtered_df["item"].nunique()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Units Sold", f"{total_sales:,.0f}")
col2.metric("Average Sales", f"{average_sales:.2f}")
col3.metric("Stores Selected", total_stores)
col4.metric("Items Selected", total_items)

st.divider()


# --------------------------------------------------
# 8. SALES TREND
# --------------------------------------------------

st.header("Sales Trends")

daily_sales = (
    filtered_df.groupby("date")["sales"]
    .sum()
    .sort_index()
)

st.markdown("**Daily Sales Trend**")
st.line_chart(daily_sales)


monthly_sales = (
    filtered_df.groupby("month")["sales"]
    .sum()
    .sort_index()
)

st.markdown("**Monthly Sales Distribution**")
st.bar_chart(monthly_sales)


# --------------------------------------------------
# 9. STORE ANALYSIS
# --------------------------------------------------

st.divider()
st.header("Store Performance Analysis")

store_sales = (
    filtered_df.groupby("store")["sales"]
    .sum()
    .sort_values(ascending=False)
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Stores")
    st.bar_chart(store_sales.head(10))

with col2:
    st.subheader("Store Sales Summary")
    st.dataframe(
        store_sales.reset_index(name="Total Sales"),
        use_container_width=True
    )


# --------------------------------------------------
# 10. ITEM ANALYSIS
# --------------------------------------------------

st.divider()
st.header("Item Demand Analysis")

item_sales = (
    filtered_df.groupby("item")["sales"]
    .sum()
    .sort_values(ascending=False)
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Items")
    st.bar_chart(item_sales.head(10))

with col2:
    st.subheader("Item Sales Summary")
    st.dataframe(
        item_sales.reset_index(name="Total Sales"),
        use_container_width=True
    )


# --------------------------------------------------
# 11. MODEL PERFORMANCE
# --------------------------------------------------

st.divider()
st.header("Machine Learning Model Performance")

st.markdown("**Baseline vs Random Forest**")

st.dataframe(
    results.round(4),
    use_container_width=True,
    hide_index=True
)

st.markdown("**Model Error Comparison**")

error_chart = results.set_index("Model")[["MAE", "RMSE"]]
st.bar_chart(error_chart)


# --------------------------------------------------
# 12. FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()
st.header("Feature Importance")

# Standardize column names for display
importance = feature_importance.copy()
importance.columns = importance.columns.str.strip().str.lower()

if {"feature", "importance"}.issubset(importance.columns):

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("feature")["importance"]
    )

    st.dataframe(
        importance,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "Feature importance file must contain "
        "'Feature' and 'Importance' columns."
    )


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "RetailPulse AI | Intelligent Retail Sales & Demand Forecasting"
)

st.caption(
    "Machine Learning Model: Random Forest Regressor"
)
