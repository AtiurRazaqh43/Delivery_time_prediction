import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeliverIQ | Delivery Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — LIGHT, MODERN DASHBOARD
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F5F7FB;
    color: #17243A;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E7ECF3;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #17243A;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #17243A;
    letter-spacing: -0.5px;
}

p, label, span {
    color: inherit;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #142B4A 0%, #203F67 65%, #E97837 150%);
    padding: 35px 38px;
    border-radius: 24px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 12px 32px rgba(26, 51, 84, 0.12);
}

.hero h1 {
    color: white !important;
    font-size: 34px;
    line-height: 1.2;
    margin-bottom: 12px;
}

.hero p {
    color: #DCE7F5 !important;
    font-size: 15px;
    line-height: 1.8;
    max-width: 700px;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    color: #FFFFFF;
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 17px;
}

/* Metric cards */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E7ECF3;
    border-radius: 18px;
    padding: 21px 23px;
    min-height: 122px;
    box-shadow: 0 4px 14px rgba(31, 49, 76, 0.035);
}

.metric-label {
    font-size: 12px;
    color: #78869A;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 12px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    color: #17243A;
}

.metric-note {
    color: #8290A3;
    font-size: 12px;
    margin-top: 7px;
}

/* Content cards */
.panel {
    background: #FFFFFF;
    border: 1px solid #E7ECF3;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 4px 14px rgba(31, 49, 76, 0.035);
    margin-bottom: 20px;
}

.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 800;
    color: #17243A;
    margin-bottom: 5px;
}

.panel-subtitle {
    font-size: 13px;
    color: #8290A3;
    margin-bottom: 22px;
}

/* Prediction result */
.result-box {
    background: linear-gradient(135deg, #FFF3E9 0%, #FFFFFF 100%);
    border: 1px solid #F6D5BC;
    border-radius: 20px;
    padding: 28px;
    margin: 14px 0 20px 0;
}

.result-label {
    color: #A85A2A;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-size: 12px;
    font-weight: 800;
}

.result-value {
    font-family: 'Manrope', sans-serif;
    font-size: 52px;
    font-weight: 800;
    color: #D96527;
    line-height: 1.25;
    margin: 10px 0;
}

.result-caption {
    color: #7D6758;
    font-size: 13px;
}

/* Buttons */
.stButton > button {
    background: #EF7C37;
    color: white;
    border: none;
    border-radius: 11px;
    padding: 0.72rem 1rem;
    font-weight: 700;
    min-height: 46px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #D96527;
    color: white;
    border: none;
    box-shadow: 0 5px 14px rgba(239, 124, 55, 0.22);
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

div[data-testid="stForm"] {
    border: none;
    padding: 0;
}

hr {
    border-color: #E7ECF3;
}

.footer {
    text-align: center;
    color: #8A96A7;
    font-size: 12px;
    padding-top: 22px;
}

.small-badge {
    background: #E9F7EF;
    color: #258653;
    border-radius: 20px;
    display: inline-block;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD THE MODEL BUNDLE
# ============================================================

MODEL_PATH = Path(__file__).parent / "food_delivery_model.pkl"


@st.cache_resource
def load_model_bundle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


if not MODEL_PATH.exists():
    st.error(
        "The file 'food_delivery_model.pkl' was not found. "
        "Place it in the same folder as app.py."
    )
    st.info(
        "Return to your Jupyter notebook, run the updated model-saving "
        "cell, and make sure the pickle file is saved in your project folder."
    )
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))

    required_keys = {
        "model",
        "encoders",
        "feature_columns",
        "categorical_columns",
        "target_column"
    }

    if not isinstance(bundle, dict) or not required_keys.issubset(bundle.keys()):
        st.error(
            "Your pickle file uses an older format. "
            "Run the updated model-saving cell in your notebook first."
        )
        st.stop()

    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_columns = bundle["feature_columns"]
    categorical_columns = bundle["categorical_columns"]

except Exception as exc:
    st.error(f"Unable to load the model bundle: {exc}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 25px 0;">
            <div style="font-size: 35px;">🛵</div>
            <div style="font-family: Manrope; font-size: 23px;
                        font-weight: 800; color: #17243A;">
                Deliver<span style="color:#EF7C37;">IQ</span>
            </div>
            <div style="font-size: 12px; color: #8290A3;">
                INTELLIGENT DELIVERY ANALYTICS
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio(
        "Choose a section",
        ["Delivery Predictor", "Model Insights"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### Model information")
    st.markdown(
        """
        **Algorithm**  
        Random Forest Regressor

        **Prediction**  
        Delivery time in minutes

        **Input processing**  
        Saved categorical encoders
        """
    )

    st.markdown("---")
    st.caption("DeliverIQ · Machine Learning Project")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">SMART LOGISTICS · AI PREDICTION</div>
        <h1>Every delivery has a story.<br>Predict how long it takes.</h1>
        <p>
            Estimate food delivery time using route distance, traffic,
            weather, preparation time and courier experience.
            Enter your delivery details to generate a model-based estimate.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DELIVERY PREDICTOR PAGE
# ============================================================

if page == "Delivery Predictor":

    st.markdown("## Delivery workspace")
    st.markdown(
        "<p style='color:#8290A3; margin-top:-8px;'>"
        "Configure the delivery conditions below and generate an estimate."
        "</p>",
        unsafe_allow_html=True
    )

    # Dashboard metrics
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Prediction engine</div>
                <div class="metric-value">Random Forest</div>
                <div class="metric-note">Ensemble regression model</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Output</div>
                <div class="metric-value">Minutes</div>
                <div class="metric-note">Estimated delivery duration</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Preprocessing</div>
                <div class="metric-value">Enabled</div>
                <div class="metric-note">Uses saved category mappings</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    left, right = st.columns([1.12, 0.88], gap="large")

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📍 Delivery details</div>
                <div class="panel-subtitle">
                    Provide the conditions for this delivery.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("delivery_form"):

            st.markdown("#### 🛣️ Route & preparation")

            col1, col2 = st.columns(2)

            with col1:
                distance = st.number_input(
                    "Distance (km)",
                    min_value=0.1,
                    max_value=200.0,
                    value=5.0,
                    step=0.5,
                    help="Approximate distance between restaurant and customer."
                )

            with col2:
                preparation_time = st.number_input(
                    "Preparation time (min)",
                    min_value=0,
                    max_value=300,
                    value=15,
                    step=1,
                    help="Time needed to prepare the order."
                )

            courier_experience = st.number_input(
                "Courier experience (years)",
                min_value=0.0,
                max_value=60.0,
                value=2.0,
                step=0.5,
                help="Approximate years of courier experience."
            )

            st.markdown("---")
            st.markdown("#### 🌦️ Delivery conditions")

            col3, col4 = st.columns(2)

            with col3:
                weather = st.selectbox(
                    "Weather",
                    options=list(encoders["Weather"].classes_),
                    help="Select the weather condition."
                )

            with col4:
                traffic = st.selectbox(
                    "Traffic level",
                    options=list(encoders["Traffic_Level"].classes_),
                    help="Select the expected traffic level."
                )

            col5, col6 = st.columns(2)

            with col5:
                time_of_day = st.selectbox(
                    "Time of day",
                    options=list(encoders["Time_of_Day"].classes_),
                    help="Select the time period of the delivery."
                )

            with col6:
                vehicle = st.selectbox(
                    "Vehicle type",
                    options=list(encoders["Vehicle_Type"].classes_),
                    help="Select the courier's vehicle."
                )

            st.markdown("")
            submitted = st.form_submit_button(
                "⚡  Predict Delivery Time",
                use_container_width=True
            )

    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    with right:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">⏱️ Prediction result</div>
                <div class="panel-subtitle">
                    Your estimated delivery duration will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if submitted:
            try:
                # Start with the original feature values.
                input_data = {
                    "Distance_km": float(distance),
                    "Weather": weather,
                    "Traffic_Level": traffic,
                    "Time_of_Day": time_of_day,
                    "Vehicle_Type": vehicle,
                    "Preparation_Time_min": float(preparation_time),
                    "Courier_Experience_yrs": float(courier_experience)
                }

                input_df = pd.DataFrame([input_data])

                # Use the exact saved encoder for each categorical column.
                for col in categorical_columns:
                    if col not in encoders:
                        raise ValueError(
                            f"No saved encoder was found for '{col}'."
                        )

                    input_df[col] = encoders[col].transform(
                        input_df[col].astype(str)
                    )

                # Match the exact feature names and order used in training.
                missing_columns = [
                    col for col in feature_columns
                    if col not in input_df.columns
                ]

                extra_columns = [
                    col for col in input_df.columns
                    if col not in feature_columns
                ]

                if missing_columns or extra_columns:
                    raise ValueError(
                        "The input columns do not match the model's training "
                        f"features. Missing: {missing_columns}; "
                        f"unexpected: {extra_columns}. Check the notebook "
                        "and saved feature_columns."
                    )

                input_df = input_df[feature_columns]

                prediction = float(model.predict(input_df)[0])

                if not pd.notna(prediction):
                    raise ValueError("The model returned an invalid prediction.")

                if prediction < 0:
                    st.warning(
                        "The model returned a negative estimate. "
                        "Please check the training data and model."
                    )
                else:
                    rounded_prediction = round(prediction, 1)

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div class="result-label">
                                Estimated delivery time
                            </div>
                            <div class="result-value">
                                {rounded_prediction}
                                <span style="font-size:22px;">min</span>
                            </div>
                            <div class="result-caption">
                                Predicted by your trained Random Forest model.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Visual timeline
                    st.markdown("#### Delivery timeline")

                    progress = min(max(prediction / 90, 0.0), 1.0)
                    st.progress(progress)

                    if prediction < 25:
                        estimate_band = "Faster estimate"
                        estimate_description = (
                            "The model predicts a relatively short delivery."
                        )
                    elif prediction < 45:
                        estimate_band = "Moderate estimate"
                        estimate_description = (
                            "The model predicts a moderate delivery duration."
                        )
                    else:
                        estimate_band = "Longer estimate"
                        estimate_description = (
                            "The model predicts a longer delivery duration."
                        )

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimate category</div>
                            <div class="metric-value"
                                 style="font-size:22px;">
                                {estimate_band}
                            </div>
                            <div class="metric-note">
                                {estimate_description}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write("")
                    st.markdown("#### Your delivery summary")

                    summary = pd.DataFrame({
                        "Delivery input": [
                            "Distance",
                            "Preparation time",
                            "Courier experience",
                            "Weather",
                            "Traffic",
                            "Time of day",
                            "Vehicle"
                        ],
                        "Selected value": [
                            f"{distance:.1f} km",
                            f"{preparation_time:.0f} min",
                            f"{courier_experience:.1f} years",
                            str(weather),
                            str(traffic),
                            str(time_of_day),
                            str(vehicle)
                        ]
                    })

                    st.dataframe(
                        summary,
                        hide_index=True,
                        use_container_width=True
                    )

                    st.caption(
                        "This is a machine-learning estimate, not a guaranteed "
                        "arrival time. Actual delivery times can vary."
                    )

            except Exception as exc:
                st.error("Prediction could not be completed.")
                st.code(str(exc))
                st.info(
                    "Confirm that the model bundle was saved after training "
                    "with the updated encoders and that its feature columns "
                    "match the input fields in this application."
                )

        else:
            st.markdown(
                """
                <div class="result-box" style="text-align:center;
                     padding:48px 24px;">
                    <div style="font-size:52px;">🛵</div>
                    <div style="font-family:Manrope; font-size:21px;
                         font-weight:800; color:#17243A; margin-top:12px;">
                        Ready when you are
                    </div>
                    <div class="result-caption" style="margin-top:10px;">
                        Complete the delivery details and click
                        <b>Predict Delivery Time</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">What you'll receive</div>
                    <div style="font-size:14px; line-height:2;
                                color:#526176;">
                        ✓ Estimated delivery duration<br>
                        ✓ Delivery input summary<br>
                        ✓ Visual estimate indicator
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# MODEL INSIGHTS PAGE
# ============================================================

elif page == "Model Insights":

    st.markdown("## Model insights")
    st.markdown(
        "<p style='color:#8290A3; margin-top:-8px;'>"
        "Explore the trained model and the features it uses."
        "</p>",
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Algorithm</div>
                <div class="metric-value" style="font-size:22px;">
                    Random Forest
                </div>
                <div class="metric-note">Ensemble learning</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Input features</div>
                <div class="metric-value">{len(feature_columns)}</div>
                <div class="metric-note">Features used by the model</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Prediction target</div>
                <div class="metric-value" style="font-size:22px;">
                    Delivery time
                </div>
                <div class="metric-note">Measured in minutes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">📊 Feature importance</div>
            <div class="panel-subtitle">
                Relative importance scores learned by the fitted model.
                Higher scores indicate greater contribution to the model's
                internal decision process; they do not prove causation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        importance_df = importance_df.set_index("Feature")

        st.bar_chart(
            importance_df,
            horizontal=True,
            height=360
        )

        st.dataframe(
            importance_df.reset_index().rename(
                columns={"Importance": "Importance score"}
            ),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info(
            "Feature-importance scores are not available for this model."
        )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">🧠 How the model works</div>
            <div style="color:#526176; line-height:1.9;">
                <b>1. Input:</b> Delivery distance, preparation time,
                courier experience and categorical delivery conditions.<br>
                <b>2. Encoding:</b> Saved encoders convert categorical
                inputs into the numerical representations used during training.<br>
                <b>3. Prediction:</b> The trained Random Forest Regressor
                estimates delivery duration in minutes.<br>
                <b>4. Output:</b> The app displays the estimated duration
                and a summary of the entered conditions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DELIVERIQ &nbsp;·&nbsp; FOOD DELIVERY TIME PREDICTION
        <br>
        Built with Streamlit, Pandas and Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)
