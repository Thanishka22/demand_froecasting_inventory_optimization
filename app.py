import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Demand Forecasting & Inventory Optimizer",
    page_icon="📦",
    layout="wide",
)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    forecasts = pd.read_csv(os.path.join(base, "sku_forecasts.csv"), parse_dates=["date"])
    policy    = pd.read_csv(os.path.join(base, "inventory_policy.csv"))
    return forecasts, policy

try:
    forecasts, policy = load_data()
    DATA_LOADED = True
    load_error = None
except Exception as e:
    DATA_LOADED = False
    load_error = str(e)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📦 Retail Demand Forecasting & Inventory Optimizer")
st.markdown(
    "An end-to-end XGBoost demand forecasting system with dynamic inventory policy simulation "
    "across multiple stores and SKUs."
)

if not DATA_LOADED:
    st.error(f"⚠️ Could not load data files. Error: {load_error}")
    st.info("Make sure `sku_forecasts.csv` and `inventory_policy.csv` are in the same folder as `app.py`.")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS & CONTROLS
# ─────────────────────────────────────────────
st.sidebar.header("🔧 Controls")

stores   = sorted(forecasts["store_key"].unique())
products = sorted(forecasts["prod_key"].unique())

selected_store   = st.sidebar.selectbox("Select Store", stores)
selected_product = st.sidebar.selectbox("Select Product (SKU)", products)

st.sidebar.divider()
st.sidebar.subheader("⚙️ What-If: Inventory Policy")

service_level = st.sidebar.select_slider(
    "Target Service Level",
    options=[0.90, 0.95, 0.97, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x*100)}%",
)

lead_time_days = st.sidebar.slider(
    "Lead Time (days)", min_value=1, max_value=21, value=7, step=1
)

holding_cost = st.sidebar.number_input(
    "Holding Cost per Unit ($/day)", min_value=0.0, value=0.50, step=0.05
)

stockout_cost = st.sidebar.number_input(
    "Stockout Cost per Unit ($)", min_value=0.0, value=5.00, step=0.25
)

# ─────────────────────────────────────────────
# FILTER DATA FOR SELECTED SKU
# ─────────────────────────────────────────────
sku_df = forecasts[
    (forecasts["store_key"] == selected_store) &
    (forecasts["prod_key"]  == selected_product)
].sort_values("date")

if sku_df.empty:
    st.warning("No forecast data found for this store/product combination.")
    st.stop()

# ─────────────────────────────────────────────
# RECALCULATE POLICY (what-if simulation)
# ─────────────────────────────────────────────
z_map = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.99: 2.33}
Z = z_map[service_level]

mu_d    = sku_df["prediction"].mean()
sigma_d = sku_df["prediction"].std() if len(sku_df) > 1 else 5.0

safety_stock  = Z * sigma_d * np.sqrt(lead_time_days)
reorder_point = mu_d * lead_time_days + safety_stock
expected_hold = safety_stock * holding_cost
expected_sout = sigma_d * 0.10 * stockout_cost
total_cost    = expected_hold + expected_sout

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
st.subheader(f"📊 SKU Summary — Store: `{selected_store}` | Product: `{selected_product}`")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Daily Forecast",    f"{mu_d:.1f} units")
col2.metric("Safety Stock",          f"{safety_stock:.0f} units")
col3.metric("Reorder Point",         f"{reorder_point:.0f} units")
col4.metric("Est. Daily Total Cost", f"${total_cost:.2f}")

st.divider()

# ─────────────────────────────────────────────
# FORECAST CHART
# ─────────────────────────────────────────────
st.subheader("📈 Demand Forecast with 95% Prediction Interval")

fig = go.Figure()

if "upper_95" in sku_df.columns and "lower_95" in sku_df.columns:
    fig.add_trace(go.Scatter(
        x=sku_df["date"], y=sku_df["upper_95"],
        mode="lines", line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=sku_df["date"], y=sku_df["lower_95"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(99,110,250,0.15)",
        name="95% Interval"
    ))

fig.add_trace(go.Scatter(
    x=sku_df["date"], y=sku_df["prediction"],
    mode="lines", line=dict(color="#636EFA", width=2),
    name="Forecast"
))

fig.add_hline(
    y=reorder_point,
    line_dash="dash", line_color="red",
    annotation_text=f"Reorder Point: {reorder_point:.0f}",
    annotation_position="top left"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Demand (units)",
    legend=dict(orientation="h", y=1.12),
    height=420,
    margin=dict(t=20, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# COST BREAKDOWN + SERVICE LEVEL TABLE
# ─────────────────────────────────────────────
st.subheader("💰 Cost Breakdown — What-If Simulation")

col_a, col_b = st.columns(2)

with col_a:
    cost_fig = go.Figure(go.Bar(
        x=["Holding Cost", "Stockout Cost", "Total Cost"],
        y=[expected_hold, expected_sout, total_cost],
        marker_color=["#00CC96", "#EF553B", "#636EFA"],
        text=[f"${expected_hold:.2f}", f"${expected_sout:.2f}", f"${total_cost:.2f}"],
        textposition="outside"
    ))
    cost_fig.update_layout(yaxis_title="Cost ($/day)", height=350, margin=dict(t=20, b=40))
    st.plotly_chart(cost_fig, use_container_width=True)

with col_b:
    rows = []
    for sl, z in z_map.items():
        ss  = z * sigma_d * np.sqrt(lead_time_days)
        rop = mu_d * lead_time_days + ss
        rows.append({"Service Level": f"{int(sl*100)}%", "Safety Stock": round(ss), "Reorder Point": round(rop)})
    st.markdown("#### Service Level Comparison")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# FULL POLICY TABLE
# ─────────────────────────────────────────────
st.divider()
st.subheader("📋 Full Inventory Policy — All SKUs")

policy_live = policy.copy()
policy_live["safety_stock"]  = Z * policy_live["sigma_d"] * np.sqrt(lead_time_days)
policy_live["reorder_point"] = policy_live["mu_d"] * lead_time_days + policy_live["safety_stock"]
policy_live["holding_cost"]  = policy_live["safety_stock"] * holding_cost
policy_live["stockout_cost"] = policy_live["sigma_d"] * 0.10 * stockout_cost
policy_live["total_cost"]    = policy_live["holding_cost"] + policy_live["stockout_cost"]

display_df = policy_live[["store_key","prod_key","mu_d","safety_stock","reorder_point","holding_cost","stockout_cost","total_cost"]].copy()
display_df.columns = ["Store","Product","Avg Daily Demand","Safety Stock","Reorder Point","Holding Cost $","Stockout Cost $","Total Cost $"]

if st.checkbox("Show selected store only", value=True):
    display_df = display_df[display_df["Store"] == selected_store]

st.dataframe(
    display_df.style.format({
        "Avg Daily Demand": "{:.1f}",
        "Safety Stock": "{:.0f}",
        "Reorder Point": "{:.0f}",
        "Holding Cost $": "${:.2f}",
        "Stockout Cost $": "${:.2f}",
        "Total Cost $": "${:.2f}",
    }),
    use_container_width=True,
    height=350,
)

st.download_button(
    label="⬇️ Download Policy as CSV",
    data=display_df.to_csv(index=False),
    file_name="inventory_policy_export.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with XGBoost · Streamlit · Plotly | Global multi-store demand forecasting")
