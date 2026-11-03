import streamlit as st
import pandas as pd
import pickle

# Website settings
st.set_page_config(
    page_title="Shopping Preference Predictor",
    page_icon="🛍️",
    layout="wide"
)
st.set_page_config(
    page_title="Shopping Preference Predictor",
    page_icon="🛍️",
    layout="wide"
)
st.set_page_config(
    page_title="Shopping Preference Predictor",
    page_icon="🛍️",
    layout="wide"
)

# PURPLE THEME
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F3E5F5, #EDE7F6);
}

[data-testid="stSidebar"] {
    background-color: #6A1B9A;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: #6A1B9A;
}

.stButton > button {
    background-color: #8E24AA;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
# Paste purple CSS code here

# Load trained machine learning model
# Load trained machine learning model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature information
with open("feature_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Sidebar navigation
st.sidebar.title("🛍️ Shopping Project")
st.sidebar.write("Customer Shopping Preference System")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Dashboard", "Prediction"]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project")

# --------------------------------
# HOME PAGE
# --------------------------------

if page == "Home":    
    
st.image("shopping_banner.jpg", use_container_width=True)
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

    st.header("About the Project")

    st.write(
        "Our project uses Machine Learning to understand "
        "customer shopping preferences. It helps identify "
        "whether customers prefer shopping online, "
        "shopping in physical stores, or both."
    )

    st.header("Problem Statement")

    st.write(
        "Understanding customer shopping preferences is "
        "important for businesses. Different customers "
        "have different shopping habits. This project "
        "uses customer information to predict their "
        "preferred shopping method."
    )

    st.header("Project Objectives")

    st.markdown("""
    - To understand customer shopping behaviour.
    - To analyse customer information.
    - To apply Machine Learning techniques.
    - To predict customer shopping preferences.
    - To support better business decisions.
    """)

    st.header("Shopping Preference Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🛒 Online")
        st.write("Shopping through websites and mobile apps.")

    with col2:
        st.subheader("🏬 Store")
        st.write("Shopping directly from physical stores.")

    with col3:
        st.subheader("🔄 Hybrid")
        st.write("Using both online and physical stores.")

    st.divider()

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

# --------------------------------
# DASHBOARD PAGE
# --------------------------------

elif page == "Dashboard":

    st.title("📊 Shopping Analytics Dashboard")

    st.write(
        "Overview of the Shopping Preference "
        "Prediction Machine Learning project."
    )

    st.divider()

    # Calculate actual project feature information
    total_features = len(metadata)

    numeric_features = sum(
        1 for info in metadata.values()
        if info["type"] == "numeric"
    )

    categorical_features = total_features - numeric_features

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Input Features", total_features)
    col2.metric("Numerical Features", numeric_features)
    col3.metric("Categorical Features", categorical_features)

    st.divider()

    st.header("Feature Analysis")

    feature_df = pd.DataFrame({
        "Feature Type": ["Numerical", "Categorical"],
        "Number of Features": [
            numeric_features,
            categorical_features
        ]
    })

    st.subheader("Numerical vs Categorical Features")

    st.bar_chart(
        feature_df.set_index("Feature Type")
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

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

    st.header("Project Information")

    st.write("*Project:* Shopping Preference Prediction")

    st.write("*Technology:* Python and Streamlit")

    st.write("*Machine Learning:* Trained prediction model")

    st.write("*Purpose:* Customer shopping preference prediction")

    st.info(
        "Note: These dashboard statistics describe the "
        "model's input features. Customer shopping trends "
        "and demographic charts can be added when the "
        "original customer dataset is available."
    )

# --------------------------------
# PREDICTION PAGE
# --------------------------------

elif page == "Prediction":

    st.title("🤖 Customer Shopping Preference Predictor")

    st.write(
        "Enter customer information to predict whether "
        "the customer prefers Online, Store, or Hybrid shopping."
    )

    st.divider()

    # Customer input form
    with st.form("customer_form"):

        customer_data = {}

        for col, info in metadata.items():

            st.subheader(col.replace("_", " ").title())

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

        input_data = pd.DataFrame([customer_data])

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Shopping Preference: {prediction}"
        )

        # Display prediction probabilities
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame({
                "Shopping Preference": model.classes_,
                "Prediction Probability": probabilities
            })

            probability_df["Prediction Probability"] *= 100

            st.subheader("Prediction Probability")

            st.dataframe(
                probability_df,
                use_container_width=True
            )

            st.bar_chart(
                probability_df.set_index("Shopping Preference")
            )

    st.divider()

    st.caption(
        "Machine Learning Based Customer Shopping Preference System"
    )
