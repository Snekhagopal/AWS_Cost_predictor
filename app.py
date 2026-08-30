import os
import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CloudCost - AWS Cost Predictor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

/* -----------------------------
GLOBAL
----------------------------- */

.stApp {
    background: #f6f9fc;
}

.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 2rem;
    max-width: 1500px;
}

h1, h2, h3, h4, p, label {
    color: #10264a !important;
}


/* -----------------------------
TOP HEADER
----------------------------- */

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #10264a;
    margin-bottom: 4px;
}

.main-title span {
    color: #1677ff;
}

.main-subtitle {
    color: #58708f;
    font-size: 16px;
    margin-bottom: 18px;
}


/* -----------------------------
SIDEBAR
----------------------------- */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #03192f 0%,
        #063761 100%
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-logo {
    font-size: 28px;
    font-weight: 800;
    margin-top: 10px;
}

.sidebar-subtitle {
    color: #c8d6e6 !important;
    margin-top: 3px;
    margin-bottom: 28px;
}

.sidebar-item {
    padding: 13px 14px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 16px;
}

.sidebar-active {
    background: linear-gradient(
        90deg,
        #1677ff,
        #246bff
    );
    font-weight: 700;
}


/* -----------------------------
CARDS
----------------------------- */

.section-card {
    background: white;
    border: 1px solid #e1e8f0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0px 4px 16px rgba(15, 23, 42, 0.06);
    margin-bottom: 14px;
}

.section-title {
    color: #10264a;
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #58708f;
    font-size: 15px;
}

.ml-badge {
    display: inline-block;
    background: #e9f2ff;
    color: #1677ff;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin-left: 8px;
    vertical-align: middle;
}


/* -----------------------------
COST CARD
----------------------------- */

.cost-card {
    background: linear-gradient(
        135deg,
        #f3fff7,
        #ecfff3
    );
    border: 1px solid #a9e6b8;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
}

.cost-label {
    color: #169342;
    font-size: 16px;
    font-weight: 700;
}

.cost-value {
    color: #18a34a;
    font-size: 50px;
    font-weight: 800;
    margin: 8px 0 2px 0;
}

.cost-small {
    color: #4a9962;
    font-size: 14px;
}

.hourly-box {
    display: inline-block;
    margin-top: 12px;
    padding: 8px 18px;
    border: 1px solid #9bd7aa;
    border-radius: 8px;
    color: #14813a;
    font-weight: 600;
}


/* -----------------------------
INFO BOX
----------------------------- */

.info-box {
    background: #eef7ff;
    border: 1px solid #a9d2ff;
    border-radius: 9px;
    padding: 14px;
    color: #173d6b;
}


/* -----------------------------
BOTTOM METRICS
----------------------------- */

.metric-card {
    background: white;
    display:flex;
    justify-content:center;
    align-items:center;
    flex-direction:column;
    border: 1px solid #e2e8f0;
    padding: 16px;
    border-radius: 10px;
    min-height: 120px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.04);
}

.metric-title {
    font-weight: 700;
    color: #10264a;
}

.metric-value {
    margin-top: 8px;
    color: #48617e;
}


/* -----------------------------
BUTTON
----------------------------- */

.stButton > button {
    width: 100%;
    height: 48px;
    background: linear-gradient(
        90deg,
        #1677ff,
        #1768e8
    );
    color: white;
    border: none;
    border-radius: 7px;
    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    color: white;
    border: none;
}


/* -----------------------------
INPUT LABELS
----------------------------- */

[data-testid="stWidgetLabel"] p {
    color: #10264a !important;
    font-weight: 600 !important;
}


/* -----------------------------
REMOVE STREAMLIT CHROME
----------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "cloudcost_model.pkl"
OPTIONS_PATH = "input_options.pkl"


if not os.path.exists(MODEL_PATH):
    st.error("cloudcost_model.pkl not found. Run training.py first.")
    st.stop()


if not os.path.exists(OPTIONS_PATH):
    st.error("input_options.pkl not found. Run training.py first.")
    st.stop()


model = joblib.load(MODEL_PATH)
input_options = joblib.load(OPTIONS_PATH)


# =========================================================
# HELPER FUNCTION
# =========================================================

def get_options(column, fallback):
    options = input_options.get(column, [])

    if not options:
        return fallback

    return options


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-logo">☁️ CloudCost</div>
<div class="sidebar-subtitle">AWS Cost Predictor</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="sidebar-item sidebar-active">
🏠 &nbsp; Dashboard
</div>

<div class="sidebar-item">
⚙️ &nbsp; EC2 Cost Estimator
</div>

<div class="sidebar-item">
📁 &nbsp; S3 Cost Estimator
</div>

<div class="sidebar-item">
🗄️ &nbsp; RDS Cost Estimator
</div>

<div class="sidebar-item">
🔖 &nbsp; Saved Estimates
</div>

<div class="sidebar-item">
🕘 &nbsp; Cost History
</div>

<div class="sidebar-item">
📊 &nbsp; Analytics
</div>

<div class="sidebar-item">
ℹ️ &nbsp; About Project
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.markdown(
        """
<div style="text-align:center;font-size:58px;">☁️</div>
<div style="text-align:center;font-size:32px;font-weight:800;">aws</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="main-title">
<span>CloudCost</span> – Predicting Cloud Infrastructure Costs
</div>

<div class="main-subtitle">
Get instant cost estimates for your AWS resources using Machine Learning
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# LAYOUT
# =========================================================

left, right = st.columns(
    [1.08, 0.92],
    gap="large"
)


# =========================================================
# LEFT SIDE
# =========================================================

with left:

    st.markdown(
        """
<div class="section-card">
<div class="section-title">
Estimate EC2 Instance Cost
<span class="ml-badge">ml-powered</span>
</div>

<div class="section-subtitle">
Configure your EC2 instance details below
</div>
</div>
""",
        unsafe_allow_html=True
    )


    # -------------------------
    # ROW 1
    # -------------------------

    c1, c2 = st.columns(2)

    with c1:
        region = st.selectbox(
            "AWS Region",
            get_options(
                "Region Code",
                ["ap-south-1"]
            )
        )

    with c2:
        instance_type = st.selectbox(
            "Instance Type",
            get_options(
                "Instance Type",
                ["m5.large"]
            )
        )


    # -------------------------
    # ROW 2
    # -------------------------

    c3, c4 = st.columns(2)

    with c3:
        operating_system = st.selectbox(
            "Operating System",
            get_options(
                "Operating System",
                ["Linux"]
            )
        )

    with c4:
        tenancy = st.selectbox(
            "Tenancy",
            get_options(
                "Tenancy",
                ["Shared"]
            )
        )


    # -------------------------
    # ROW 3
    # -------------------------

    c5, c6 = st.columns(2)

    with c5:
        vcpu = st.number_input(
            "vCPUs",
            min_value=1.0,
            value=2.0,
            step=1.0
        )

    with c6:
        memory = st.number_input(
            "Memory (GB)",
            min_value=0.0,
            value=8.0,
            step=1.0
        )


    # -------------------------
    # ROW 4
    # -------------------------

    c7, c8 = st.columns(2)

    with c7:
        storage_type = st.selectbox(
            "Storage Type",
            get_options(
                "Storage_Type",
                ["EBS only"]
            )
        )

    with c8:
        total_storage = st.number_input(
            "EBS Storage (GB)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )


    # -------------------------
    # ROW 5
    # -------------------------

    c9, c10 = st.columns(2)

    with c9:
        term_type = st.selectbox(
            "Usage Type",
            get_options(
                "TermType",
                ["OnDemand"]
            )
        )

    with c10:
        usage_hours = st.number_input(
            "Usage (Hours)",
            min_value=1.0,
            value=730.0,
            step=1.0
        )


    # =====================================================
    # ADVANCED INPUTS
    # =====================================================

    with st.expander("Advanced Configuration"):

        a1, a2 = st.columns(2)

        with a1:

            clock_speed = st.number_input(
                "Clock Speed (GHz)",
                min_value=0.0,
                value=3.0
            )

            gpu = st.number_input(
                "GPU Count",
                min_value=0.0,
                value=0.0
            )

            gpu_memory = st.number_input(
                "GPU Memory (GB)",
                min_value=0.0,
                value=0.0
            )

            normalization_factor = st.number_input(
                "Normalization Size Factor",
                min_value=0.0,
                value=1.0
            )

            storage_count = st.number_input(
                "Storage Count",
                min_value=0.0,
                value=0.0
            )

            ebs_throughput = st.number_input(
                "EBS Throughput (Mbps)",
                min_value=0.0,
                value=0.0
            )

            instance_family = st.selectbox(
                "Instance Family",
                get_options(
                    "Instance Family",
                    ["General purpose"]
                )
            )

            processor_arch = st.selectbox(
                "Processor Architecture",
                get_options(
                    "Processor Architecture",
                    ["64-bit"]
                )
            )


        with a2:

            current_generation = st.selectbox(
                "Current Generation",
                get_options(
                    "Current Generation",
                    ["Yes"]
                )
            )

            license_model = st.selectbox(
                "License Model",
                get_options(
                    "License Model",
                    ["No License required"]
                )
            )

            capacity_status = st.selectbox(
                "Capacity Status",
                get_options(
                    "CapacityStatus",
                    ["Used"]
                )
            )

            preinstalled_software = st.selectbox(
                "Pre Installed Software",
                get_options(
                    "Pre Installed S/W",
                    ["Not Applicable"]
                )
            )

            purchase_option = st.selectbox(
                "Purchase Option",
                get_options(
                    "PurchaseOption",
                    ["Not Applicable"]
                )
            )

            lease_contract = st.selectbox(
                "Lease Contract Length",
                get_options(
                    "LeaseContractLength",
                    ["Not Applicable"]
                )
            )

            offering_class = st.selectbox(
                "Offering Class",
                get_options(
                    "OfferingClass",
                    ["Not Applicable"]
                )
            )


    predict_button = st.button(
        "▣ Predict Cost"
    )


    st.markdown(
        """
<div class="info-box">
💡 This cost is an estimate generated by our ML model trained on AWS pricing data.
Actual AWS pricing may vary.
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION
# =========================================================

predicted_hourly_price = 0.0
estimated_cost = 0.0


if predict_button:

    input_data = pd.DataFrame(
        {
            "vCPU": [vcpu],
            "Memory_GiB": [memory],
            "ClockSpeed_GHz": [clock_speed],
            "GPU": [gpu],
            "GPU_Memory_GB": [gpu_memory],

            "Normalization Size Factor": [
                normalization_factor
            ],

            "Storage_Count": [
                storage_count
            ],

            "Total_Storage_GB": [
                total_storage
            ],

            "Storage_Type": [
                storage_type
            ],

            "EBS_Throughput_Mbps": [
                ebs_throughput
            ],

            "Instance Type": [
                instance_type
            ],

            "Instance Family": [
                instance_family
            ],

            "Processor Architecture": [
                processor_arch
            ],

            "Current Generation": [
                current_generation
            ],

            "Region Code": [
                region
            ],

            "Operating System": [
                operating_system
            ],

            "Tenancy": [
                tenancy
            ],

            "License Model": [
                license_model
            ],

            "CapacityStatus": [
                capacity_status
            ],

            "Pre Installed S/W": [
                preinstalled_software
            ],

            "TermType": [
                term_type
            ],

            "PurchaseOption": [
                purchase_option
            ],

            "LeaseContractLength": [
                lease_contract
            ],

            "OfferingClass": [
                offering_class
            ]
        }
    )


    predicted_hourly_price = float(
        model.predict(input_data)[0]
    )

    estimated_cost = (
        predicted_hourly_price
        * usage_hours
    )


# =========================================================
# RIGHT SIDE
# =========================================================

with right:

    st.markdown(
        """
<div class="section-card">
<div class="section-title">
Estimated Cost
</div>
</div>
""",
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
<div class="cost-card">
<div class="cost-label">
Predicted Estimated Cost
</div>

<div class="cost-value">
${estimated_cost:,.2f}
</div>

<div class="cost-small">
USD
</div>

<div class="hourly-box">
${predicted_hourly_price:,.4f} / hour
</div>
</div>
""",
        unsafe_allow_html=True
    )


    st.markdown("### Breakdown")


    breakdown_df = pd.DataFrame(
        {
            "Item": [
                "EC2 Instance Cost",
                f"Usage ({usage_hours:.0f} hours)",
                "Total Estimated Cost"
            ],

            "Estimated Cost (USD)": [
                f"${predicted_hourly_price:.4f} / hour",
                f"{usage_hours:.0f} hours",
                f"${estimated_cost:,.2f}"
            ]
        }
    )


    st.dataframe(
        breakdown_df,
        hide_index=True,
        use_container_width=True
    )


    st.markdown(
        f"""
<div class="info-box">
<b>⚙ Configuration Summary</b>
<br><br>

<table style="width:100%;color:#173d6b;">
<tr>
<td>Region</td>
<td><b>{region}</b></td>
</tr>

<tr>
<td>Instance Type</td>
<td><b>{instance_type}</b></td>
</tr>

<tr>
<td>OS</td>
<td><b>{operating_system}</b></td>
</tr>

<tr>
<td>Tenancy</td>
<td><b>{tenancy}</b></td>
</tr>

<tr>
<td>Usage</td>
<td><b>{usage_hours:.0f} hours</b></td>
</tr>
</table>
</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# MODEL METRICS
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)


m1, m2, m3, m4, m5 = st.columns(5)


with m1:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-title">
🧠 Model Used
</div>

<div class="metric-value">
Random Forest Regressor
</div>
</div>
""",
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-title">
📈 R² Score
</div>

<div class="metric-value">
0.990
</div>
</div>
""",
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-title">
🎯 MAE
</div>

<div class="metric-value">
0.460 USD
</div>
</div>
""",
        unsafe_allow_html=True
    )


with m4:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-title">
〽️ RMSE
</div>

<div class="metric-value">
2.193 USD
</div>
</div>
""",
        unsafe_allow_html=True
    )


with m5:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-title">
🗄️ Trained On
</div>

<div class="metric-value">
AWS Pricing Dataset
</div>
</div>
""",
        unsafe_allow_html=True
    )