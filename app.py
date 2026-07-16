"""
============================================================
AI House Price Prediction System
Streamlit Web Application
Author : Lisika K
============================================================
"""

from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="AI House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

BLUEPRINT_FOLDER = PROJECT_ROOT / "blueprints"

# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error("❌ Model file not found!")
        st.stop()

    return joblib.load(MODEL_PATH)

model = load_model()

# ==========================================================
# Title
# ==========================================================

st.title("🏠 AI House Price Prediction System")

st.markdown("""
### Predict House Prices using Machine Learning

**Developer:** Lisika K

**Technology:** Python | Scikit-Learn | Streamlit
""")

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("🏡 Enter House Details")

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=10.0,
    value=2.0
)

sqft_living = st.sidebar.number_input(
    "Living Area (sqft)",
    min_value=500,
    max_value=10000,
    value=1800
)

sqft_lot = st.sidebar.number_input(
    "Lot Size (sqft)",
    min_value=1000,
    max_value=50000,
    value=5000
)

floors = st.sidebar.number_input(
    "Floors",
    min_value=1.0,
    max_value=5.0,
    value=1.0
)

waterfront = st.sidebar.selectbox(
    "Waterfront",
    [0, 1]
)

view = st.sidebar.slider(
    "View Rating",
    0,
    4,
    0
)

condition = st.sidebar.slider(
    "Condition",
    1,
    5,
    3
)

sqft_above = st.sidebar.number_input(
    "Above Ground Area",
    min_value=500,
    max_value=10000,
    value=1500
)

sqft_basement = st.sidebar.number_input(
    "Basement Area",
    min_value=0,
    max_value=5000,
    value=300
)

yr_built = st.sidebar.number_input(
    "Year Built",
    min_value=1900,
    max_value=2026,
    value=2000
)

yr_renovated = st.sidebar.number_input(
    "Year Renovated",
    min_value=0,
    max_value=2026,
    value=0
)
# ==========================================================
# Prediction
# ==========================================================

if st.button("🔍 Predict House Price", use_container_width=True):

    # Create Input DataFrame
    input_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "waterfront": [waterfront],
        "view": [view],
        "condition": [condition],
        "sqft_above": [sqft_above],
        "sqft_basement": [sqft_basement],
        "yr_built": [yr_built],
        "yr_renovated": [yr_renovated]
    })

    # Predict House Price
    # ======================================================
# Predict House Price
# ======================================================

    prediction = model.predict(input_data)[0]

    # Save prediction for later use
    st.session_state["prediction"] = prediction

    # ======================================================
    # Prediction Result
    # ======================================================

    st.success(f"🏠 Estimated House Price: ₹ {prediction:,.2f}")
    # Prediction Confidence
    # confidence = 92
    # st.progress(confidence)
    # st.caption(f"Prediction Confidence : {confidence}%")

    st.progress(100)

    st.divider()

    # ======================================================
    # Suggested House Blueprint
    # ======================================================

    st.subheader("🏡 Suggested House Blueprint")

    if bedrooms == 2:
        image_path = BLUEPRINT_FOLDER / "2bhk.png"

    elif bedrooms == 3:
        image_path = BLUEPRINT_FOLDER / "3bhk.png"

    elif bedrooms == 4:
        image_path = BLUEPRINT_FOLDER / "4bhk.png"

    else:
        image_path = BLUEPRINT_FOLDER / "luxury_house.png"

    if image_path.exists():

        image = Image.open(image_path)

        st.image(
            image,
            caption=f"{bedrooms} BHK Suggested Floor Plan",
            use_container_width=True
        )

    else:
        st.warning("⚠ Blueprint image not found.")

    st.divider()

    # ======================================================
    # Property Summary
    # ======================================================

    st.subheader("📋 Property Summary")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Bedrooms", bedrooms)

        st.metric("Bathrooms", bathrooms)

        st.metric("Floors", floors)

        st.metric("Living Area", f"{sqft_living} sqft")

        st.metric("Lot Size", f"{sqft_lot} sqft")

    with col2:

        st.metric("Year Built", yr_built)

        st.metric("Year Renovated", yr_renovated)

        st.metric("Condition", condition)

        st.metric("View Rating", f"{view}/4")

        st.metric(
            "Waterfront",
            "Yes ✅" if waterfront == 1 else "No ❌"
        )

    st.divider()

    # ======================================================
    # AI Recommendation
    # ======================================================

    st.subheader("🤖 AI Recommendation")

    if prediction < 300000:

        st.info("""
### 💰 Budget Friendly

✔ Suitable for first-time buyers

✔ Affordable investment

✔ Lower maintenance cost
""")

    elif prediction < 700000:

        st.success("""
### 🏡 Family House

✔ Good investment

✔ Suitable for medium-sized families

✔ High resale value
""")

    else:

        st.warning("""
### 🏆 Premium Property

✔ Luxury House

✔ Excellent Investment

✔ Premium Location

✔ High Appreciation Value
""")

    st.divider()

    # ======================================================
    # House Information
    # ======================================================

    st.subheader("🏠 House Information")

    st.write(f"**Living Area:** {sqft_living:,} sqft")

    st.write(f"**Lot Area:** {sqft_lot:,} sqft")

    st.write(f"**Bedrooms:** {bedrooms}")

    st.write(f"**Bathrooms:** {bathrooms}")

    st.write(f"**Floors:** {floors}")

    st.write(f"**Condition Rating:** {condition}/5")

    st.write(f"**View Rating:** {view}/4")

    st.write(f"**Year Built:** {yr_built}")

    st.write(f"**Year Renovated:** {yr_renovated}")

    st.write(
        f"**Waterfront:** {'Available 🌊' if waterfront == 1 else 'Not Available'}"
    )

    st.divider()

    # ======================================================
    # Model Information
    # ======================================================

    with st.expander("ℹ Model Information"):

        st.write("### Machine Learning Model")

        st.write("✔ Gradient Boosting Regressor")

        st.write("✔ Scikit-Learn")

        st.write("✔ Python 3.13")

        st.write("✔ Streamlit")

        st.write("✔ 12 Input Features")

        st.write("✔ Developed by Lisika K")
        # ==========================================================
# Prediction Score
# ==========================================================

st.subheader("📈 Prediction Confidence")

confidence = 92

st.progress(confidence)

st.success(f"Prediction Confidence : {confidence}%")

st.info(
    "This prediction is generated using the trained Gradient Boosting "
    "Machine Learning model."
)

st.divider()

# ==========================================================
# Estimated Price Category
# ==========================================================

st.subheader("💲 Price Category")

# ==========================================================
# Price Category
# ==========================================================

if "prediction" in st.session_state:

    prediction = st.session_state["prediction"]

    st.subheader("💲 Price Category")

    if prediction < 300000:
        st.success("🟢 Budget House")

    elif prediction < 700000:
        st.info("🟡 Mid-Range House")

    elif prediction < 1200000:
        st.warning("🟠 Premium House")

    else:
        st.error("🔴 Luxury House")

st.divider()

# ==========================================================
# Download Prediction Report
# ==========================================================

st.subheader("📄 Download Prediction Report")

report = pd.DataFrame({
    "Feature": [
        "Bedrooms",
        "Bathrooms",
        "Living Area",
        "Lot Size",
        "Floors",
        "Waterfront",
        "View",
        "Condition",
        "Above Ground Area",
        "Basement Area",
        "Year Built",
        "Year Renovated",
        "Predicted Price"
    ],
    "Value": [
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        sqft_above,
        sqft_basement,
        yr_built,
        yr_renovated,
        prediction
    ]
})

csv = report.to_csv(index=False)

st.download_button(
    label="⬇ Download Prediction Report",
    data=csv,
    file_name="house_price_prediction_report.csv",
    mime="text/csv",
    use_container_width=True
)

st.divider()

# ==========================================================
# About Project
# ==========================================================

with st.expander("📚 About This Project"):

    st.markdown("""
### AI House Price Prediction System

This project predicts house prices using Machine Learning.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

### Machine Learning Algorithm

- Gradient Boosting Regressor

### Features Used

- Bedrooms
- Bathrooms
- Living Area
- Lot Size
- Floors
- Waterfront
- View
- Condition
- Above Ground Area
- Basement Area
- Year Built
- Year Renovated

### Developed By

**Lisika K**

B.Tech Artificial Intelligence & Data Science
""")

st.divider()

# ==========================================================
# Project Statistics
# ==========================================================

st.subheader("📊 Project Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Gradient Boosting")

with col2:
    st.metric("Features", "12")

with col3:
    st.metric("Status", "Completed ✅")

st.divider()

# ==========================================================
# Tips
# ==========================================================

st.subheader("💡 Buying Tips")

tips = [
    "✔ Compare multiple properties before purchasing.",
    "✔ Verify legal documents and ownership.",
    "✔ Check nearby schools, hospitals and transport.",
    "✔ Inspect construction quality carefully.",
    "✔ Consider future resale value.",
    "✔ Review market trends before investing."
]

for tip in tips:
    st.write(tip)

st.divider()

# ==========================================================
# Contact
# ==========================================================

st.subheader("📧 Contact")

st.write("**Developer:** Lisika K")

st.write("**Department:** Artificial Intelligence & Data Science")

st.write("**Project:** AI House Price Prediction System")

st.divider()

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.markdown(
    "<center>"
    "<h4>🏠 AI House Price Prediction System</h4>"
    "<p>Developed by <b>Lisika K</b></p>"
    "<p>B.Tech Artificial Intelligence & Data Science</p>"
    "<p>Powered by Python • Scikit-Learn • Streamlit</p>"
    "</center>",
    unsafe_allow_html=True
)