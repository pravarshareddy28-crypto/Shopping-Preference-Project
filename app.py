import streamlit as st
import pandas as pd
import pickle
import os

# --------------------------------
# WEBSITE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Shopping Preference Predictor",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------
# LIGHT LAVENDER + PURPLE THEME
# --------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #F7F2FC,
        #EEE6F7
    );
    color: #222222;
}

/* Main content text */
.stApp p,
.stApp label,
.stApp span,
.stApp li,
.stApp div {
    color: #222222;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #51217A !important;
    font-weight: 700;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #E5D5F2;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #241331 !important;
}

/* Sidebar heading */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #51217A !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #6A1B9A !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
}

/* Button hover */
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #4A126F !important;
    color: white !important;
    border: none;
}

/* Input boxes */
.stTextInput input,
.stNumberInput input {
    background-color: white !important;
    color: #222222 !important;
    border-radius: 8px;
}

/* Dropdowns */
[data-baseweb="select"] {
    background-color: white;
}

/* Cards and containers */
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #D9C5EA;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #C9A9E0;
}

/* Responsive layout */
@media screen and (max-width: 768px) {
    h1 {
        font-size: 26px !important;
    }

    h2 {
        font-size: 22px !important;
    }

    h3 {
        font-size: 19px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------

try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("feature_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

except FileNotFoundError as e:
    st.error(
        f"Required project file not found: {e.filename}. "
        "Please check that best_model.pkl and "
        "feature_metadata.pkl are uploaded to your GitHub repository."
    )
    st.stop()


# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("🛍️ Shopping Project")

st.sidebar.write(
    "Customer Shopping Preference System"
)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Dashboard", "Prediction"]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Project"
)


# ================================================
# HOME PAGE
# ================================================

if page == "Home":

    st.title("🛍️ Shopping Preference Prediction")
    st.subheader("Welcome to Our Project")

    st.write(
        "A Machine Learning Based Customer Shopping "
        "Preference Prediction System."
    )

    st.info(
        "This project analyses customer information "
        "to predict whether a customer prefers Online, "
        "Store, or Hybrid shopping."
    )

    st.divider()

    # About
    st.header("About the Project")

    st.write(
        "Our project uses Machine Learning to understand "
        "customer shopping preferences. It helps identify "
        "whether customers prefer shopping online, "
        "shopping in physical stores, or both."
    )

    # Problem Statement
    st.header("Problem Statement")

    st.write(
        "Understanding customer shopping preferences is "
        "important for businesses. Different customers "
        "have different shopping habits. This project "
        "uses customer information to predict their "
        "preferred shopping method."
    )

    # Objectives
    st.header("Project Objectives")

    st.markdown("""
    - To understand customer shopping behaviour.
    - To analyse customer information.
    - To apply Machine Learning techniques.
    - To predict customer shopping preferences.
    - To support better business decisions.
    """)

    # Shopping categories
    st.header("Shopping Preference Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🛒 Online")
        st.write(
            "Shopping through websites and mobile apps."
        )

    with col2:
        st.subheader("🏬 Store")
        st.write(
            "Shopping directly from physical stores."
        )

    with col3:
        st.subheader("🔄 Hybrid")
        st.write(
            "Using both online and physical stores."
        )

    st.divider()

    # Machine Learning
    st.header("Machine Learning")

    st.write(
        "A trained Machine Learning model is used "
        "to predict customer shopping preferences "
        "based on the entered information."
    )

    st.success(
        "Explore the Dashboard and Prediction pages "
        "using the navigation menu."
    )


# ================================================
# DASHBOARD PAGE
# ================================================

elif page == "Dashboard":

    st.title("📊 Shopping Analytics Dashboard")

    st.write(
        "Overview of the Shopping Preference "
        "Prediction Machine Learning project."
    )

    st.divider()

    # Calculate actual feature information
    total_features = len(metadata)

    numeric_features = sum(
        1
        for info in metadata.values()
        if info["type"] == "numeric"
    )

    categorical_features = (
        total_features - numeric_features
    )

    # Dashboard metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Input Features",
        total_features
    )

    col2.metric(
        "Numerical Features",
        numeric_features
    )

    col3.metric(
        "Categorical Features",
        categorical_features
    )

    st.divider()

    # Feature analysis
    st.header("Feature Analysis")

    feature_df = pd.DataFrame({
        "Feature Type": [
            "Numerical",
            "Categorical"
        ],
        "Number of Features": [
            numeric_features,
            categorical_features
        ]
    })

    st.subheader(
        "Numerical vs Categorical Features"
    )

    st.bar_chart(
        feature_df.set_index("Feature Type")
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Model classes
    st.header("Shopping Preference Categories")

    if hasattr(model, "classes_"):

        classes_df = pd.DataFrame({
            "Shopping Preference": model.classes_
        })

        st.write(
            "The trained model supports the following "
            "shopping preference categories:"
        )

        st.dataframe(
            classes_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # Project information
    st.header("Project Information")

    st.write(
        "*Project:* Shopping Preference Prediction"
    )

    st.write(
        "*Technology:* Python and Streamlit"
    )

    st.write(
        "*Machine Learning:* Trained prediction model"
    )

    st.write(
        "*Purpose:* Customer shopping preference prediction"
    )

    st.info(
        "Note: These dashboard statistics describe the "
        "model's input features. Customer shopping trends "
        "and demographic charts can be added when the "
        "original customer dataset is available."
    )


# ================================================
# PREDICTION PAGE
# ================================================

elif page == "Prediction":

    st.title(
        "🤖 Customer Shopping Preference Predictor"
    )

    st.write(
        "Enter customer information to predict whether "
        "the customer prefers Online, Store, or Hybrid shopping."
    )

    st.divider()

    # Customer input form
    with st.form("customer_form"):

        customer_data = {}

        for col, info in metadata.items():

            st.subheader(
                col.replace("_", " ").title()
            )

            if info["type"] == "numeric":

                if info["integer"]:

                    customer_data[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=int(info["min"]),
                        max_value=int(info["max"]),
                        value=int(info["default"])
                    )

                else:

                    customer_data[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=float(info["min"]),
                        max_value=float(info["max"]),
                        value=float(info["default"])
                    )

            else:

                options = info["options"]
                default = info["default"]

                customer_data[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options=options,
                    index=options.index(default)
                )

        submitted = st.form_submit_button(
            "Predict Shopping Preference",
            use_container_width=True
        )

    # Prediction result
    if submitted:

        try:

            input_data = pd.DataFrame(
                [customer_data]
            )

            prediction = model.predict(
                input_data
            )[0]

            st.success(
                f"Predicted Shopping Preference: {prediction}"
            )

            # Prediction probabilities
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                probability_df = pd.DataFrame({
                    "Shopping Preference": model.classes_,
                    "Prediction Probability": probabilities
                })

                probability_df[
                    "Prediction Probability"
                ] *= 100

                st.subheader(
                    "Prediction Probability"
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    probability_df.set_index(
                        "Shopping Preference"
                    )
                )

        except Exception as e:

            st.error(
                f"Unable to generate prediction: {e}"
            )

    st.divider()

    st.caption(
        "Machine Learning Based Customer Shopping Preference System"
    )
