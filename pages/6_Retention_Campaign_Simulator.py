
import streamlit as st

st.set_page_config(
    page_title="Retention Campaign Simulator",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Retention Campaign Simulator")

st.write(
    "Estimate the financial impact of retention campaigns "
    "for customers who may leave the company."
)

st.info(
    "This simulator uses business assumptions. "
    "Its estimates are not guaranteed revenue or actual campaign results."
)

st.subheader("1. Campaign Configuration")

col1, col2 = st.columns(2)

with col1:
    customers = st.number_input(
        "Number of high-risk customers",
        min_value=1,
        max_value=100000,
        value=100,
        step=10
    )

    monthly_revenue = st.number_input(
        "Average monthly revenue per customer ($)",
        min_value=1.0,
        value=70.0,
        step=5.0
    )

with col2:
    discount = st.number_input(
        "One-time discount per customer ($)",
        min_value=0.0,
        value=20.0,
        step=5.0
    )

    retention_rate = st.slider(
        "Assumed campaign retention rate (%)",
        min_value=0,
        max_value=100,
        value=30
    )

retention_months = st.slider(
    "Expected additional months retained",
    min_value=1,
    max_value=36,
    value=6
)

st.divider()

st.subheader("2. Campaign Results")

# Calculate campaign estimates
retained_customers = round(
    customers * retention_rate / 100
)

campaign_cost = customers * discount

revenue_saved = (
    retained_customers
    * monthly_revenue
    * retention_months
)

net_benefit = revenue_saved - campaign_cost

roi = (
    (net_benefit / campaign_cost) * 100
    if campaign_cost > 0
    else None
)

# Display metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Estimated customers retained",
    f"{retained_customers:,}"
)

col2.metric(
    "Campaign cost",
    f"${campaign_cost:,.2f}"
)

col3.metric(
    "Potential revenue retained",
    f"${revenue_saved:,.2f}"
)

st.divider()

col1, col2 = st.columns(2)

col1.metric(
    "Estimated net benefit",
    f"${net_benefit:,.2f}",
    delta=f"${net_benefit:,.2f}"
)

if roi is not None:
    col2.metric("Estimated campaign ROI", f"{roi:.2f}%")
else:
    col2.metric("Estimated campaign ROI", "Not applicable")

st.subheader("3. Business Recommendation")

if net_benefit > 0:
    st.success(
        "Under the selected assumptions, the campaign "
        "has a positive estimated financial benefit."
    )
elif net_benefit < 0:
    st.warning(
        "Under the selected assumptions, the campaign "
        "cost is higher than the estimated retained revenue."
    )
else:
    st.info(
        "The estimated campaign benefit is equal to its cost."
    )

st.subheader("4. Calculation Details")

st.write(f"- Customers selected: {customers:,}")
st.write(f"- Assumed retention rate: {retention_rate}%")
st.write(f"- Estimated retained customers: {retained_customers:,}")
st.write(f"- Discount per customer: ${discount:,.2f}")
st.write(f"- Expected retention period: {retention_months} months")

st.caption(
    "Campaign cost = Selected customers × One-time discount. "
    "Potential revenue retained = Estimated retained customers × "
    "Monthly revenue × Additional months retained. "
    "Net benefit = Potential revenue retained − Campaign cost. "
    "ROI = Net benefit ÷ Campaign cost × 100. "
    "These calculations use gross revenue, not profit, and assume "
    "the retention rate and retention period are accurate."
)
