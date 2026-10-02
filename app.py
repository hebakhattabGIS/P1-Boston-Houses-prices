import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ----------------------------------------------------------------------
# CONFIG - adjust these to match your project
# ----------------------------------------------------------------------
MODEL_PATH = r"Model\model.pkl"          # your saved model (joblib or pickle)
DATA_PATH = r"Data\housing.csv"         # Boston housing CSV (used only for the charts)
FEATURES = ["RM", "LSTAT", "PTRATIO"]  # MUST match the order used in training
TARGET = "MEDV"                   # target column in the CSV
PRICE_SCALE = 1                   # housing.csv already stores MEDV in dollars
# ----------------------------------------------------------------------

st.set_page_config(page_title="Boston Housing Price Prediction", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df = df[FEATURES + [TARGET]].dropna()
        df["Price"] = df[TARGET] * PRICE_SCALE
        return df
    except Exception:
        return None


model = load_model()
data = load_data()

# ---------------------------- Sidebar ---------------------------------
st.sidebar.title("🏠 Home Features")

rm = st.sidebar.slider("Average number of rooms (RM)", 3.0, 9.0, 6.0, 0.1)
lstat = st.sidebar.slider("Neighborhood poverty level (LSTAT %)", 1.0, 40.0, 12.0, 0.5)
ptratio = st.sidebar.slider("Student-teacher ratio (PTRATIO)", 10.0, 25.0, 18.0, 0.5)

if st.sidebar.button("Predict Home Price", type="primary", use_container_width=True):
    X = pd.DataFrame([[rm, lstat, ptratio]], columns=FEATURES)
    pred = float(model.predict(X)[0]) * PRICE_SCALE
    st.session_state["prediction"] = pred
    st.session_state["inputs"] = {"RM": rm, "LSTAT": lstat, "PTRATIO": ptratio}

# ----------------------------- Main -----------------------------------
st.title("Boston Housing Price Prediction Dashboard")
st.write(
    "This application uses a Decision Tree Regressor to estimate home prices "
    "based on user inputs. Adjust the features in the sidebar and click **Predict Home Price**."
)

if "prediction" in st.session_state:
    st.success("Prediction ready")
    st.markdown("##### Predicted Selling Price")
    st.markdown(
        f"<h1 style='color:#1b9e77;margin-top:-10px'>${st.session_state['prediction']:,.2f}</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Value is an estimate based on model training data.")
else:
    st.info("Set the features in the sidebar and click **Predict Home Price**.")

# ----------------------------- Charts ---------------------------------
st.subheader("Explore Key Features")

if data is None:
    st.warning(f"Could not load '{DATA_PATH}', so the charts are hidden. Check DATA_PATH / column names in the config.")
else:
    def scatter(feature, title, xlabel):
        base = (
            alt.Chart(data)
            .mark_circle(size=40, opacity=0.5)
            .encode(
                x=alt.X(f"{feature}:Q", title=xlabel, scale=alt.Scale(zero=False)),
                y=alt.Y("Price:Q", title="Price ($)"),
                tooltip=[feature, "Price"],
            )
            .properties(title=title, height=300)
        )
        if "prediction" in st.session_state:
            pt = pd.DataFrame(
                {feature: [st.session_state["inputs"][feature]], "Price": [st.session_state["prediction"]]}
            )
            marker = (
                alt.Chart(pt)
                .mark_point(size=250, color="red", filled=True, shape="diamond")
                .encode(x=f"{feature}:Q", y="Price:Q")
            )
            return base + marker
        return base

    c1, c2 = st.columns(2)
    c1.altair_chart(scatter("RM", "Average Rooms vs Price", "Rooms (RM)"), use_container_width=True)
    c2.altair_chart(scatter("LSTAT", "Poverty Rate vs Price", "Poverty % (LSTAT)"), use_container_width=True)
    if "prediction" in st.session_state:
        st.caption("The red diamond marks your prediction.")

# ------------------------- Feature info -------------------------------
st.subheader("Feature Information")
st.markdown(
    """
- **RM**: average number of rooms per dwelling
- **LSTAT**: percentage of lower-status population in the neighborhood
- **PTRATIO**: pupil-teacher ratio by town
"""
)
st.caption("Built with Streamlit and scikit-learn | Data source: Boston Housing Dataset")
