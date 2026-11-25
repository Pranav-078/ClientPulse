
import streamlit as st
import pandas as pd
from datetime import datetime, timezone

from src.data_loader import load_data
from src.train import load_best_model, load_model_metadata
from src.preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.predict import (
    validate_customer_input,
    classify_risk_tier,
    DEFAULT_RISK_THRESHOLDS
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Risk Prioritization | ClientPulse",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Customer Risk Prioritization")

st.caption(
    "Batch customer scoring, risk segmentation and "
    "revenue exposure analysis."
)

st.info(
    "Dataset notice: This version scores the historical IBM Telco dataset. "
    "It demonstrates batch inference and prioritization, not live customer "
    "monitoring. Historical records may have been used during model training."
)


# =====================================================
# 2. LOAD PROJECT RESOURCES
# =====================================================

@st.cache_data
def get_dataset():
    return load_data()


@st.cache_resource
def get_trained_model():
    return load_best_model()


@st.cache_data
def get_model_information():
    return load_model_metadata()


try:
    data = get_dataset()
    model = get_trained_model()
    metadata = get_model_information()

except Exception as error:
    st.error(f"Unable to initialize ClientPulse: {error}")
    st.exception(error)
    st.stop()


if model is None:
    st.error(
        "Trained model not found. Please train the model first."
    )

    st.code("python -m src.train", language="powershell")
    st.stop()


if not hasattr(model, "predict_proba"):
    st.error(
        "The loaded model does not support probability prediction. "
        "A calibrated probability-capable classifier is required."
    )
    st.stop()


# =====================================================
# 3. VALIDATE DATA AND MODEL FEATURES
# =====================================================

required_columns = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(
        f"Dataset is missing required model features: {missing_columns}"
    )
    st.stop()


if "customerID" not in data.columns:
    st.error("Customer ID column is missing.")
    st.stop()


if "MonthlyCharges" not in data.columns:
    st.error("MonthlyCharges column is missing.")
    st.stop()


# =====================================================
# 4. BATCH PREDICTION ENGINE
# =====================================================

@st.cache_data(show_spinner="Scoring customer accounts...")
def generate_customer_scores(
    customer_data,
    _model
):

    # Preserve identifiers separately.
    customer_ids = customer_data["customerID"].astype(str)

    # Only use features expected by the existing ML pipeline.
    # Never pass customerID or actual Churn labels.
    model_features = customer_data[
        required_columns
    ].copy()

    # Apply the same input validation used by single prediction.
    clean_features = validate_customer_input(
        model_features
    )

    # Find the probability column corresponding to class 1 (Churn).
    model_classes = list(_model.classes_)

    if 1 not in model_classes:
        raise ValueError(
            "The trained model does not contain class 1 (Churn)."
        )

    churn_index = model_classes.index(1)

    probabilities = _model.predict_proba(
        clean_features
    )[:, churn_index]

    results = pd.DataFrame({
        "Customer ID": customer_ids.values,
        "Churn Probability": probabilities,
        "Monthly Charges": pd.to_numeric(
            customer_data["MonthlyCharges"],
            errors="coerce"
        ).values,
        "Contract": customer_data["Contract"].values,
        "Tenure": customer_data["tenure"].values,
        "Internet Service": customer_data["InternetService"].values,
        "Payment Method": customer_data["PaymentMethod"].values
    })

    if results["Churn Probability"].isna().any():
        raise ValueError("Invalid churn probabilities generated.")

    if not results["Churn Probability"].between(0, 1).all():
        raise ValueError("Model generated probabilities outside [0, 1].")

    # Reuse the existing project's risk classification.
    risk_results = results["Churn Probability"].apply(
        classify_risk_tier
    )

    results["Risk Level"] = risk_results.apply(
        lambda result: result[0]
    )

    # Probability-weighted monthly revenue exposure.
    results["Expected Monthly Revenue Exposure"] = (
        results["Churn Probability"] *
        results["Monthly Charges"]
    )

    # Priority ranking based on revenue exposure.
    results = results.sort_values(
        "Expected Monthly Revenue Exposure",
        ascending=False
    ).reset_index(drop=True)

    results.insert(
        0,
        "Priority Rank",
        range(1, len(results) + 1)
    )

    return results


try:
    scored_customers = generate_customer_scores(
        data,
        model
    )

except Exception as error:
    st.error("Customer scoring failed.")
    st.exception(error)
    st.stop()


# =====================================================
# 5. MODEL INFORMATION
# =====================================================

with st.expander("Model Information", expanded=False):

    if metadata:
        st.write(
            "**Model:**",
            metadata.get("model_name", "Not available")
        )

        st.write(
            "**Training timestamp:**",
            metadata.get("training_timestamp", "Not available")
        )

        metrics = metadata.get("metrics", {})

        if metrics:
            st.write("**Training evaluation metadata:**")
            st.json(metrics)

    else:
        st.warning(
            "Model metadata file is unavailable."
        )


# =====================================================
# 6. EXECUTIVE OVERVIEW
# =====================================================

st.divider()
st.subheader("Executive Risk Overview")

high_risk = scored_customers[
    scored_customers["Risk Level"] == "High Risk"
]

moderate_risk = scored_customers[
    scored_customers["Risk Level"] == "Moderate Risk"
]

low_risk = scored_customers[
    scored_customers["Risk Level"] == "Low Risk"
]

total_customers = len(scored_customers)

total_exposure = scored_customers[
    "Expected Monthly Revenue Exposure"
].sum()

high_risk_exposure = high_risk[
    "Expected Monthly Revenue Exposure"
].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers Scored",
    f"{total_customers:,}"
)

col2.metric(
    "High-Risk Customers",
    f"{len(high_risk):,}"
)

col3.metric(
    "Moderate-Risk Customers",
    f"{len(moderate_risk):,}"
)

col4.metric(
    "High-Risk Monthly Exposure",
    f"${high_risk_exposure:,.2f}"
)

st.caption(
    f"Total probability-weighted monthly revenue exposure: "
    f"${total_exposure:,.2f}"
)


# =====================================================
# 7. RISK DISTRIBUTION
# =====================================================

st.divider()
st.subheader("Risk Distribution")

risk_summary = pd.DataFrame({
    "Risk Level": [
        "High Risk",
        "Moderate Risk",
        "Low Risk"
    ],
    "Customers": [
        len(high_risk),
        len(moderate_risk),
        len(low_risk)
    ]
})

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("**Customer distribution**")

    st.bar_chart(
        risk_summary.set_index("Risk Level")
    )

with chart_col2:
    st.write("**Risk-level percentages**")

    risk_summary["Percentage"] = (
        risk_summary["Customers"] /
        total_customers * 100
    )

    st.dataframe(
        risk_summary,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 8. CUSTOMER FILTERS
# =====================================================

st.divider()
st.subheader("Customer Prioritization")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    selected_risks = st.multiselect(
        "Risk classification",
        ["High Risk", "Moderate Risk", "Low Risk"],
        default=["High Risk", "Moderate Risk", "Low Risk"]
    )

with filter_col2:
    contract_options = ["All"] + sorted(
        scored_customers["Contract"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_contract = st.selectbox(
        "Contract type",
        contract_options
    )

with filter_col3:
    internet_options = ["All"] + sorted(
        scored_customers["Internet Service"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_internet = st.selectbox(
        "Internet service",
        internet_options
    )


search_id = st.text_input(
    "Search customer ID",
    placeholder="Enter customer ID..."
)

minimum_probability = st.slider(
    "Minimum churn probability (%)",
    min_value=0,
    max_value=100,
    value=0
)


# =====================================================
# 9. APPLY FILTERS
# =====================================================

filtered = scored_customers[
    scored_customers["Risk Level"].isin(selected_risks)
].copy()

if selected_contract != "All":
    filtered = filtered[
        filtered["Contract"] == selected_contract
    ]

if selected_internet != "All":
    filtered = filtered[
        filtered["Internet Service"] == selected_internet
    ]

filtered = filtered[
    filtered["Churn Probability"] >= (
        minimum_probability / 100
    )
]

if search_id.strip():
    filtered = filtered[
        filtered["Customer ID"].str.contains(
            search_id.strip(),
            case=False,
            na=False
        )
    ]

filtered = filtered.sort_values(
    "Expected Monthly Revenue Exposure",
    ascending=False
).reset_index(drop=True)

filtered["Priority Rank"] = range(
    1,
    len(filtered) + 1
)


# =====================================================
# 10. PRIORITIZED CUSTOMER TABLE
# =====================================================

st.subheader("Prioritized Customer List")

st.caption(
    f"{len(filtered):,} customers match your filters."
)

display_columns = [
    "Priority Rank",
    "Customer ID",
    "Risk Level",
    "Churn Probability",
    "Monthly Charges",
    "Expected Monthly Revenue Exposure",
    "Contract",
    "Tenure"
]

if filtered.empty:

    st.warning(
        "No customers match your selected filters."
    )

else:

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={

            "Churn Probability":
                st.column_config.ProgressColumn(
                    "Churn Probability",
                    min_value=0,
                    max_value=1,
                    format="percent"
                ),

            "Monthly Charges":
                st.column_config.NumberColumn(
                    "Monthly Charges",
                    format="$%.2f"
                ),

            "Expected Monthly Revenue Exposure":
                st.column_config.NumberColumn(
                    "Expected Monthly Exposure",
                    format="$%.2f"
                )
        }
    )


# =====================================================
# 11. TOP PRIORITY CUSTOMERS
# =====================================================

st.divider()
st.subheader("Top 10 Customers by Revenue Exposure")

top_customers = filtered.head(10)

if not top_customers.empty:

    chart_data = top_customers.set_index(
        "Customer ID"
    )[["Expected Monthly Revenue Exposure"]]

    st.bar_chart(chart_data)

else:
    st.info("No customers available for this chart.")


# =====================================================
# 12. INDIVIDUAL CUSTOMER PROFILE
# =====================================================

st.divider()
st.subheader("Individual Customer Risk Profile")

if not filtered.empty:

    selected_customer_id = st.selectbox(
        "Select customer",
        filtered["Customer ID"].tolist()
    )

    customer = filtered[
        filtered["Customer ID"] == selected_customer_id
    ].iloc[0]

    profile_col1, profile_col2, profile_col3 = st.columns(3)

    profile_col1.metric(
        "Churn Probability",
        f"{customer['Churn Probability']:.1%}"
    )

    profile_col2.metric(
        "Risk Classification",
        customer["Risk Level"]
    )

    profile_col3.metric(
        "Monthly Revenue Exposure",
        f"${customer['Expected Monthly Revenue Exposure']:.2f}"
    )

    st.write("**Customer Details**")

    st.json({
        "Customer ID": customer["Customer ID"],
        "Contract": customer["Contract"],
        "Tenure (Months)": int(customer["Tenure"]),
        "Internet Service": customer["Internet Service"],
        "Payment Method": customer["Payment Method"],
        "Monthly Charges": float(customer["Monthly Charges"])
    })

    st.caption(
        "The displayed probability is a model estimate. "
        "It does not guarantee that this customer will churn."
    )


# =====================================================
# 13. EXPORT REPORT
# =====================================================

st.divider()
st.subheader("Export Business Report")

export_columns = display_columns

export_data = filtered[export_columns].copy()

export_data["Scoring Timestamp UTC"] = (
    datetime.now(timezone.utc).isoformat()
)

csv_file = export_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Customer Risk Report",
    data=csv_file,
    file_name="clientpulse_customer_risk_report.csv",
    mime="text/csv",
    disabled=filtered.empty
)


# =====================================================
# 14. BUSINESS GOVERNANCE
# =====================================================

with st.expander("Scoring Methodology and Limitations"):

    st.markdown(
        """
        **Risk classification**

        - Low Risk: probability below 35%
        - Moderate Risk: probability from 35% to below 60%
        - High Risk: probability of 60% or higher

        These are the existing operational thresholds in
        `src/predict.py`.

        **Priority calculation**

        Expected Monthly Revenue Exposure =
        Churn Probability × Monthly Charges

        This is a probability-weighted revenue exposure indicator.
        It is not confirmed lost revenue, profit, or customer lifetime value.

        **Data limitations**

        - The IBM dataset is historical and cross-sectional.
        - The current page scores historical records, including records
          that may have appeared in model training.
        - The current page is not a live production customer feed.
        - Prediction scores are estimates, not causal explanations.
        - Retention actions should be evaluated using controlled experiments.
        """
    )

st.caption(
    "ClientPulse | Customer Churn Intelligence Platform"
)
