import streamlit as st
import pandas as pd
import pickle

# Load the trained machine learning model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature information
with open("feature_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Website settings
st.set_page_config(
    page_title="Shopping Preference Predictor",
    page_icon="🛍️",
    layout="centered"
)

# Website heading
st.title("🛍️ Customer Shopping Preference Predictor")

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
