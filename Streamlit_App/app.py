import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AURA OS - From CSV to Shaadi", layout="wide", page_icon="💍")

# ---------------------------
# CSS Theme: Gen-Z Investor UI
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 25%, #1f2937 60%, #0b1327 100%);
        color: #f8fafc;
    }
    h1, h2, h3, h4 {
        color: #f8fafc;
    }
    .stTabs [role="tablist"] button {
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .stTabs [role="tablist"] .st-emotion-cache-1r6slb0 {
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
    }
    div[data-testid="stMetricValue"] {
        color: #a7f3d0;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Demo Data Generation
# ---------------------------

def build_demo_orders():
    rng = pd.date_range("2024-01-01", "2026-12-31", periods=500)
    categories = ["Furniture", "Office Supplies", "Technology"]
    regions = ["West", "East", "Central", "South"]
    segments = ["Consumer", "Corporate", "Home Office"]
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    products = {
        "Furniture": ["Office Chair", "Desk", "Bookshelf", "Table", "Cabinet"],
        "Office Supplies": ["Notebook", "Printer Ink", "Files", "Stapler", "Marker Set"],
        "Technology": ["Laptop", "Monitor", "Keyboard", "Mouse", "Router"],
    }
    rows = []
    for i in range(500):
        category = categories[i % len(categories)]
        region = regions[i % len(regions)]
        segment = segments[i % len(segments)]
        ship_mode = ship_modes[i % len(ship_modes)]
        product = products[category][i % len(products[category])]
        sales = round(1500 + (i * 17.5) % 9000, 2)
        discount = round((i % 10) * 0.05, 2)
        quantity = 1 + (i % 6)
        profit = round(sales * (0.12 + (i % 7) * 0.02) - (discount * sales) - (i % 5) * 22, 2)
        rows.append(
            {
                "Order ID": f"ORD-{i + 1000}",
                "Order Date": rng[i],
                "Customer ID": f"CUST-{1000 + i}",
                "Customer Name": f"Customer {i % 80}",
                "Segment": segment,
                "Country": "United States",
                "City": f"City {i % 25}",
                "State": f"State {i % 15}",
                "Postal Code": 10000 + i,
                "Region": region,
                "Product ID": f"PROD-{1000 + i}",
                "Category": category,
                "Sub-Category": product,
                "Product Name": product,
                "Sales": sales,
                "Quantity": quantity,
                "Discount": discount,
                "Profit": profit,
                "Ship Mode": ship_mode,
                "Shipping Cost": round((i % 10) * 25 + 30, 2),
            }
        )
    return pd.DataFrame(rows)


def build_demo_returns(orders_df):
    return_rows = []
    for idx, row in orders_df.iterrows():
        if idx % 12 == 0:
            return_rows.append(
                {
                    "Order ID": row["Order ID"],
                    "Return Date": row["Order Date"] + pd.Timedelta(days=7),
                    "Product Name": row["Product Name"],
                    "Return Qty": 1,
                    "Return Amount": round(max(0.0, abs(row["Profit"]) * 0.35), 2),
                    "Reason": "High Discount" if row["Discount"] > 0.2 else "High Shipping",
                    "Customer Name": row["Customer Name"],
                }
            )
    return pd.DataFrame(return_rows)


def build_demo_people():
    return pd.DataFrame(
        {
            "Name": ["Riya", "Aman", "Shreya", "Rahul", "Neha"],
            "Role": ["Sales Lead", "Ops Head", "Returns Manager", "Account Lead", "Customer Success"],
            "Last Purchase Date": pd.to_datetime(["2024-03-15", "2024-01-20", "2024-04-10", "2024-02-01", "2024-05-05"]),
            "Spend": [12000, 8000, 9500, 11000, 14000],
        }
    )


def load_uploaded_csv(uploaded_file):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


@st.cache_data
def load_data():
    orders_file = st.session_state.get("orders_upload")
    returns_file = st.session_state.get("returns_upload")
    people_file = st.session_state.get("people_upload")

    if orders_file is not None:
        orders = load_uploaded_csv(orders_file)
    else:
        orders = build_demo_orders()

    if returns_file is not None:
        returns = load_uploaded_csv(returns_file)
    else:
        returns = build_demo_returns(orders)

    if people_file is not None:
        people = load_uploaded_csv(people_file)
    else:
        people = build_demo_people()

    for df in [orders, returns, people]:
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]

    if "Order ID" in orders.columns and "Order ID" in returns.columns:
        merged = orders.merge(returns[["Order ID", "Return Qty", "Return Amount", "Reason", "Return Date"]], on="Order ID", how="left")
    else:
        merged = orders.copy()
        merged["Return Qty"] = 0
        merged["Return Amount"] = 0.0
        merged["Reason"] = "No Return"
        merged["Return Date"] = pd.NaT

    if "Sales" in merged.columns:
        merged["Sales_INR"] = merged["Sales"] * 83
    if "Profit" in merged.columns:
        merged["Profit_INR"] = merged["Profit"] * 83

    return merged, returns, people


# ---------------------------
# App state / upload widgets
# ---------------------------

st.sidebar.title("AURA OS")
st.sidebar.markdown("### CSV Injector")
st.sidebar.caption("Professional Name: Intelligent Data Ingestion System")
st.sidebar.caption("Pronunciation note: See-Ess-Vee In-jec-tor")

st.sidebar.file_uploader("Upload Orders.csv", type=["csv"], key="orders_upload")
st.sidebar.file_uploader("Upload Returns.csv", type=["csv"], key="returns_upload")
st.sidebar.file_uploader("Upload People.csv", type=["csv"], key="people_upload")

# auto-load data on app startup
orders_df, returns_df, people_df = load_data()

# Sidebar KPI / filters
st.sidebar.header("Live Filters")
if "Category" in orders_df.columns:
    category_filter = st.sidebar.multiselect("Category", sorted(orders_df["Category"].unique().tolist()), default=sorted(orders_df["Category"].unique().tolist()))
    orders_df = orders_df[orders_df["Category"].isin(category_filter)]

if "Region" in orders_df.columns:
    region_filter = st.sidebar.multiselect("Region", sorted(orders_df["Region"].unique().tolist()), default=sorted(orders_df["Region"].unique().tolist()))
    orders_df = orders_df[orders_df["Region"].isin(region_filter)]

# ---------------------------
# Helper functions for analytics
# ---------------------------

def format_rupees(x):
    return f"₹{x:,.2f}"


def get_aura_recommendations(profit_inr, loss_inr, return_loss_inr):
    # Safe API key fetch - no crash if secrets.toml missing
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""
    profit_value = float(profit_inr) if pd.notna(profit_inr) else 0.0
    loss_value = float(loss_inr) if pd.notna(loss_inr) else 0.0
    return_value = float(return_loss_inr) if pd.notna(return_loss_inr) else 0.0

    prompt = (
        "You are a senior retail strategy advisor. "
        "Based on the following business data, provide exactly 3 bullet points only in professional English. "
        "Each bullet should be a practical action that a retail business should take next. "
        "Focus on profit protection, pricing, inventory, return leakage, and strategic growth. "
        f"Profit in INR: {profit_value:.2f}. Loss in INR: {loss_value:.2f}. Return-related loss in INR: {return_value:.2f}. "
        "Return the answer as plain text with each bullet starting with '- '."
    )

    if not api_key:
        return [
            "Rework discounting and shipping policy to reduce avoidable margin leakage and protect category profitability.",
            "Prioritize inventory for high-converting categories and reduce slow-moving SKUs to improve cash efficiency.",
            "Build a tighter returns and supplier review loop to prevent repeated product and service failures from damaging margin.",
        ]

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a sharp retail strategy advisor."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 250,
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        bullets = [line.strip() for line in content.splitlines() if line.strip().startswith("-")]
        if len(bullets) >= 3:
            return [b[1:].strip() for b in bullets[:3]]
    except Exception:
        pass

    return [
        "Reduce discount-driven margin erosion and align pricing with category profitability before expanding promotional activity.",
        "Tighten inventory planning around the strongest demand segments and stop overstocking low-converting products.",
        "Improve returns recovery and supplier accountability to reduce product leakage and restore retail margin quality.",
    ]


def safe_month_col(df):
    if "Order Date" in df.columns:
        df = df.copy()
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
        df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


orders_df = safe_month_col(orders_df)
if "Profit_INR" not in orders_df.columns:
    orders_df["Profit_INR"] = orders_df.get("Profit", 0) * 83
if "Sales_INR" not in orders_df.columns:
    orders_df["Sales_INR"] = orders_df.get("Sales", 0) * 83

# ---------------------------
# Header / title
# ---------------------------
st.title("🧠 AURA OS - From CSV to Shaadi: An End-to-End AI Powered Retail Operating System")
st.caption("Investor-ready retail intelligence suite for demand, profit, returns, customer experience, and wedding commerce.")
st.markdown("---")

# ---------------------------
# Tab setup
# ---------------------------
tab_dashboard, tab_profit_lab, tab_ai_future, tab_addiction_lab, tab_wedding_os = st.tabs([
    "DASHBOARD",
    "PROFIT LAB",
    "AI FUTURE TELLER",
    "ADDICTION LAB",
    "WEDDING OS",
])

# ---------------------------
# DASHBOARD Tab
# ---------------------------
with tab_dashboard:
    st.subheader("📊 DASHBOARD")
    total_sales = orders_df["Sales_INR"].sum() if "Sales_INR" in orders_df.columns else 0
    total_profit = orders_df["Profit_INR"].sum() if "Profit_INR" in orders_df.columns else 0
    total_loss = max(0, -total_profit)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", format_rupees(total_sales))
    c2.metric("Total Profit", format_rupees(total_profit))
    c3.metric("Total Loss", format_rupees(total_loss))

    monthly = orders_df.groupby("Month", as_index=False)["Profit_INR"].sum() if "Month" in orders_df.columns else pd.DataFrame({"Month": [], "Profit_INR": []})
    if not monthly.empty:
        fig = px.line(monthly, x="Month", y="Profit_INR", markers=True, title="Monthly Profit Trend in INR")
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Monthly trend not available yet.")

    st.markdown("### P&L Center")
    st.caption("Pee-and-Ell Center")

    c4, c5 = st.columns(2)
    with c4:
        category_profit = orders_df.groupby("Category", as_index=False)["Profit_INR"].sum().sort_values("Profit_INR", ascending=False)
        fig2 = px.bar(category_profit, x="Category", y="Profit_INR", color="Category", title="Profit by Category")
        st.plotly_chart(fig2, use_container_width=True)
    with c5:
        region_sales = orders_df.groupby("Region", as_index=False)["Sales_INR"].sum().sort_values("Sales_INR", ascending=False)
        fig3 = px.pie(region_sales, names="Region", values="Sales_INR", title="Sales by Region")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Stock Siren")
    st.caption("Stock Sai-ren")
    stock_df = orders_df.groupby("Product Name", as_index=False)["Quantity"].sum().sort_values("Quantity")
    stock_alerts = stock_df[stock_df["Quantity"] < 10]
    if stock_alerts.empty:
        st.success("All products are above minimum stock threshold. No immediate reorder required.")
    else:
        for _, row in stock_alerts.head(5).iterrows():
            st.error(f"REORDER NOW: {row['Product Name']} - Only {int(row['Quantity'])} Left")

# ---------------------------
# PROFIT LAB Tab
# ---------------------------
with tab_profit_lab:
    st.subheader("💸 PROFIT LAB")

    st.markdown("### Loss Tracker")
    st.caption("Loss Attribution Engine")
    loss_df = orders_df[orders_df["Profit_INR"] < 0].copy()
    if loss_df.empty:
        st.info("No loss-making transactions found in this dataset.")
    else:
        loss_df["Reason"] = np.where(loss_df["Discount"] > 0.2, "High Discount", "High Shipping")
        st.dataframe(loss_df[["Order ID", "Product Name", "Category", "Profit_INR", "Discount", "Shipping Cost", "Reason"]], use_container_width=True)

    st.markdown("### Profit Finder")
    st.caption("Profitability Driver Analysis")
    best_category = orders_df.groupby("Category", as_index=False)["Profit_INR"].sum().sort_values("Profit_INR", ascending=False).head(1)
    best_month = orders_df.groupby("Month", as_index=False)["Profit_INR"].sum().sort_values("Profit_INR", ascending=False).head(1)
    c1, c2 = st.columns(2)
    c1.metric("Best Category", best_category["Category"].iloc[0])
    c2.metric("Best Month", best_month["Month"].iloc[0])

    st.markdown("### Return Radar")
    st.caption("Return Ray-dar")
    if returns_df.empty:
        st.warning("No returns data found. Using demo returns pipeline.")
    returns_summary = returns_df.copy()
    if "Return Amount" in returns_summary.columns:
        returns_summary["Return Amount"] = pd.to_numeric(returns_summary["Return Amount"], errors="coerce").fillna(0)
    if "Product Name" in returns_summary.columns:
        top_returns = returns_summary.groupby("Product Name").agg(Count=("Product Name", "size"), Loss=("Return Amount", "sum")).reset_index().sort_values(["Count", "Loss"], ascending=False).head(10)
        st.dataframe(top_returns, use_container_width=True)
        order_total = orders_df.shape[0]
        return_rate = returns_summary.shape[0] / order_total if order_total else 0
        if return_rate > 0.10:
            st.warning("Supplier Badlo")
    else:
        st.info("Returns file is missing product information for radar analysis.")

    st.markdown("### Blame Report")
    st.caption("Blame Re-port")
    blame_df = pd.DataFrame()
    if not loss_df.empty:
        blame_df = loss_df[["Order Date", "Profit_INR", "Reason", "Customer Name"]].copy()
        blame_df["Responsible Person"] = np.where(blame_df["Reason"] == "High Discount", "Sales Lead", "Logistics Head")
        blame_df["Action Required"] = np.where(blame_df["Reason"] == "High Discount", "Rework discount pricing", "Audit shipment mode and freight")
        blame_df = blame_df.rename(columns={"Order Date": "Date", "Profit_INR": "Loss/Profit", "Customer Name": "Responsible Person"})
        blame_df["Date"] = pd.to_datetime(blame_df["Date"]).dt.strftime("%Y-%m-%d")
        blame_df = blame_df[["Date", "Loss/Profit", "Reason", "Responsible Person", "Action Required"]]
        for _, row in blame_df.head(5).iterrows():
            st.error(f"{row['Date']} | {format_rupees(row['Loss/Profit'])} | {row['Reason']} | {row['Responsible Person']} | {row['Action Required']}")
    else:
        st.success("No blame report required. Profitability is healthy.")

# ---------------------------
# AI FUTURE TELLER Tab
# ---------------------------
with tab_ai_future:
    st.subheader("🔮 AI FUTURE TELLER")

    st.markdown("### Demand Pulse")
    st.caption("Dee-mand Pulse")
    product_sales = orders_df.groupby("Product Name", as_index=False)["Sales_INR"].sum().sort_values("Sales_INR", ascending=False)
    product_stock = orders_df.groupby("Product Name", as_index=False)["Quantity"].sum()
    trend_df = product_sales.merge(product_stock, on="Product Name")
    trend_df["sales_to_stock_ratio"] = trend_df["Sales_INR"] / trend_df["Quantity"].replace(0, 1)
    high_demand = trend_df[trend_df["sales_to_stock_ratio"] > trend_df["sales_to_stock_ratio"].quantile(0.75)]
    if not high_demand.empty:
        st.success("Stock Bharo - Demand Badhne Wali Hai")
        st.dataframe(high_demand.head(5), use_container_width=True)
    else:
        st.info("Demand is stable; no immediate stock stress detected.")

    st.markdown("### Festival Radar")
    st.caption("Fes-ti-val Ray-dar")
    sales_by_month = orders_df.groupby("Month", as_index=False)["Sales_INR"].sum().sort_values("Sales_INR", ascending=False)
    peak_month = sales_by_month.iloc[0] if not sales_by_month.empty else None
    if peak_month is not None:
        st.info(f"Peak sales month: {peak_month['Month']} with {format_rupees(peak_month['Sales_INR'])}")
        st.success("Diwali/Shaadi season pehle se stock lo")
    else:
        st.info("No month-level sales trend available.")

    st.markdown("### Profit Forecast")
    st.caption("Revenue Projection System - 90 Day")
    if "Month" in orders_df.columns:
        month_profit = orders_df.groupby("Month", as_index=False)["Profit_INR"].sum().sort_values("Month")
        month_profit = month_profit.tail(12).reset_index(drop=True)
        if len(month_profit) >= 2:
            X = np.arange(len(month_profit)).reshape(-1, 1)
            y = month_profit["Profit_INR"].values
            model = LinearRegression()
            model.fit(X, y)
            future_steps = np.arange(len(month_profit), len(month_profit) + 3).reshape(-1, 1)
            future_preds = model.predict(future_steps)
            future_df = pd.DataFrame({"Month": [f"Future-{i}" for i in range(1, 4)], "Profit_INR": future_preds})
            fig4 = px.line(pd.concat([month_profit.assign(Month=month_profit["Month"].astype(str)), future_df], ignore_index=True), x="Month", y="Profit_INR", markers=True)
            fig4.add_scatter(x=future_df["Month"], y=future_df["Profit_INR"], mode="lines+markers", line=dict(dash="dot", color="orange"), name="Forecast")
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Forecasting not available without monthly data.")

    st.markdown("### Aura Says")
    st.caption("Aura Strategic Insights")
    total_profit = orders_df["Profit_INR"].sum() if "Profit_INR" in orders_df.columns else 0
    total_loss = max(0, -(orders_df["Profit_INR"].sum() if "Profit_INR" in orders_df.columns else 0))
    return_loss = 0
    if returns_df is not None and "Return Amount" in returns_df.columns:
        return_loss = pd.to_numeric(returns_df["Return Amount"], errors="coerce").fillna(0).sum() * 83

    aura_points = get_aura_recommendations(total_profit, total_loss, return_loss)
    st.markdown("**Aage kya karna chahiye**")
    for point in aura_points:
        st.markdown(f"- {point}")

    st.markdown("### Churn Alert")
    st.caption("Churn A-lert")
    if "Customer Name" in orders_df.columns and "Order Date" in orders_df.columns:
        last_purchase = orders_df.groupby("Customer Name", as_index=False)["Order Date"].max().rename(columns={"Order Date": "Last Date"})
        spend = orders_df.groupby("Customer Name", as_index=False)["Sales_INR"].sum().rename(columns={"Sales_INR": "Spend"})
        churn = last_purchase.merge(spend, on="Customer Name")
        churn["Last Date"] = pd.to_datetime(churn["Last Date"])
        churn["Days Since Last Purchase"] = (pd.Timestamp.today().normalize() - churn["Last Date"]).dt.days
        churn_alert = churn[churn["Days Since Last Purchase"] > 90].head(10)
        if churn_alert.empty:
            st.success("Customer retention is strong. No churn risk identified.")
        else:
            churn_alert = churn_alert[["Customer Name", "Last Date", "Spend", "Days Since Last Purchase"]].copy()
            churn_alert["Action"] = "20% Coupon Bhejo"
            st.dataframe(churn_alert, use_container_width=True)

# ---------------------------
# ADDICTION LAB Tab
# ---------------------------
with tab_addiction_lab:
    st.subheader("🎯 ADDICTION LAB")

    st.markdown("### Flash Engine")
    st.caption("Flash En-gine")
    if "Order Date" in orders_df.columns and "Product Name" in orders_df.columns:
        max_date = orders_df["Order Date"].max()
        slow = orders_df.groupby("Product Name", as_index=False)["Order Date"].max()
        slow = slow[slow["Order Date"] < max_date - pd.Timedelta(days=30)]
        if slow.empty:
            st.success("No slow-moving products detected. The catalog is moving briskly.")
        else:
            for _, row in slow.head(3).iterrows():
                st.error(f"Aaj Ka Dhamaka 40% OFF - Timer 02:00:00 | {row['Product Name']}")

    st.markdown("### Spin & Win")
    st.caption("Gamified Rewards & Loyalty System")
    if st.button("Chakri Ghumao - 10-50% Jeeto"):
        discount = np.random.randint(10, 51)
        st.balloons()
        st.success(f"Lucky spin result: {discount}% off on your next order!")
        st.markdown(f"<div class='card'>Mystery Card: {discount}% discount unlocked</div>", unsafe_allow_html=True)
    else:
        st.info("Spin the wheel to unlock a mystery reward.")

    st.markdown("### Style Lab")
    st.caption("Style Lab")
    style_df = orders_df.copy()
    style_df["Style Type"] = style_df["Product Name"].fillna("").str.lower().apply(
        lambda x: "Checked" if "checked" in x else "Plain" if "plain" in x else "Printed" if "printed" in x else "Oversized" if "oversized" in x else "Other"
    )
    style_summary = style_df["Style Type"].value_counts().reset_index()
    style_summary.columns = ["Style Type", "Count"]
    style_summary["Share"] = style_summary["Count"] / style_summary["Count"].sum() * 100
    if not style_summary.empty:
        fig5 = px.pie(style_summary, names="Style Type", values="Count", title="Style Mix")
        st.plotly_chart(fig5, use_container_width=True)
        top_style = style_summary.sort_values("Count", ascending=False).iloc[0]
        st.success(f"{top_style['Style Type']} {top_style['Share']:.0f}% bika")

    st.markdown("### Price Lens")
    st.caption("Visual Commerce & Competitive Price Intelligence")
    uploaded_image = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Product Image", use_container_width=True)
    comparison = pd.DataFrame(
        {
            "Platform": ["Myntra", "Meesho", "AURA"],
            "Price": [1299, 799, 899],
        }
    )
    st.dataframe(comparison, use_container_width=True)
    st.success("Best Deal: AURA ₹899")

# ---------------------------
# WEDDING OS Tab
# ---------------------------
with tab_wedding_os:
    st.subheader("💍 WEDDING OS")

    if "crowd_ideas" not in st.session_state:
        st.session_state.crowd_ideas = []
    if "wedding_bookings" not in st.session_state:
        st.session_state.wedding_bookings = []

    st.markdown("### Crowd Hub")
    st.caption("Crowd Hub")
    with st.form("crowd_form"):
        name = st.text_input("Name")
        design_idea = st.text_input("Design Idea")
        color = st.text_input("Color")
        budget = st.number_input("Budget", min_value=0, step=1000)
        submitted = st.form_submit_button("Submit Idea")

    if submitted:
        st.session_state.crowd_ideas.append({"Name": name, "Design Idea": design_idea, "Color": color, "Budget": budget})
        crowd_df = pd.DataFrame(st.session_state.crowd_ideas)
        if not crowd_df.empty:
            design_counts = crowd_df["Design Idea"].value_counts()
            if (design_counts > 10).any():
                st.warning("100 log maang rahe hain - Banao to bikega!")
        st.success("Idea saved to Crowd Hub.")

    if st.session_state.crowd_ideas:
        st.dataframe(pd.DataFrame(st.session_state.crowd_ideas), use_container_width=True)

    st.markdown("### Wedding OS")
    st.caption("Wed-ding O-S")
    with st.form("wedding_booking_form"):
        shaadi_code = st.text_input("Shaadi Code (example: RAMESH-WEDS-2026)")
        member_name = st.text_input("Member Name")
        product = st.text_input("Product")
        qty = st.number_input("Qty", min_value=1, step=1)
        delivery_date = st.date_input("Delivery Date")
        wedding_submitted = st.form_submit_button("Book Product")

    if wedding_submitted:
        entry = {
            "Shaadi Code": shaadi_code,
            "Member Name": member_name,
            "Product": product,
            "Qty": qty,
            "Delivery Date": str(delivery_date),
            "Cost": float(qty * 1500),
        }
        st.session_state.wedding_bookings.append(entry)
        st.success("Wedding booking captured successfully.")

    if st.session_state.wedding_bookings:
        booking_df = pd.DataFrame(st.session_state.wedding_bookings)
        st.dataframe(booking_df, use_container_width=True)
        col_total_members, col_total_cost, col_delivery = st.columns(3)
        col_total_members.metric("Total Members", booking_df["Member Name"].nunique())
        col_total_cost.metric("Total Cost", format_rupees(float(booking_df["Cost"].sum())))
        should_single_day = booking_df["Delivery Date"].nunique() <= 1
        if should_single_day:
            st.success("Single date delivery promise is maintained.")
        else:
            st.info("Delivery dates are spread across multiple dates.")

    st.markdown("### Anti-Clash")
    st.caption("An-ti Clash")
    with st.form("anti_clash_form"):
        anti_code = st.text_input("Shaadi Code")
        anti_member = st.text_input("Member Name")
        anti_product = st.text_input("Product")
        anti_color = st.text_input("Color")
        anti_submit = st.form_submit_button("Check Booking")

    if anti_submit:
        existing = pd.DataFrame(st.session_state.wedding_bookings)
        if not existing.empty:
            same_design = (
                (existing["Shaadi Code"] == anti_code)
                & (existing["Product"].astype(str).str.lower() == anti_product.lower())
                & (existing["Member Name"].astype(str).str.lower() != anti_member.lower())
            )
            if same_design.any():
                st.error("BLOCKED! Ye Design Already Booked Hai - No Same-Same! Photo me alag lagoge")
            else:
                st.success("Booking approved. Design is unique for this shaadi code.")
        else:
            st.success("No existing booking found. Approved.")

st.markdown("---")
st.caption("Built for AURA OS • Final Year Project • Investor-ready Retail Operating System")
