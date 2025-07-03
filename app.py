import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_model():
    if not os.path.exists("lgbm.pkl"):
        st.stop()  # Stop if missing
    return joblib.load("lgbm.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
preprocessor = load_preprocessor()
label_encoder = load_label_encoder()

# List of games
all_games = [
    'Rocket', 'Simple', 'Divided', 'Birds', 'Habitats', 'Snake', 'Targets', 'Sams Garden',
    'Ladybug', 'Ski', 'Cloudy', 'Lumina', 'Hexagon', 'Evolve', 'Flexi', 'Drops', 'Simon',
    'React', 'Arrows', 'Flaneur'
]

# Initialize session state
if "show_results" not in st.session_state:
    st.session_state.show_results = False

st.title("Game Accessibility Classifier")

# --- Page 1: Form ---
if not st.session_state.show_results:
    st.markdown("Please fill out the form below:")

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
            Age = st.number_input("Age", min_value=18, max_value=100)
            care_level = st.number_input("Care Level (0-12)", min_value=0, max_value=12)
            BMI = st.number_input("BMI", min_value=10.0, max_value=50.0)
            education_level = st.number_input("Education Level", min_value=0, max_value=30)

        with col2:
            Sex = st.selectbox("Sex", ["Select...", "Male", "Female"])
            mobility_type = st.selectbox("Mobility Type", ["Select...", "No", "Yes"])
            previous_experience = st.selectbox("Previous Gaming Experience", ["Select...", "Yes", "No"])

        st.subheader("Cognitive & Physical Values")
        col3, col4 = st.columns(2)
        with col3:
            sppb = st.number_input("SPPB (0–12)", min_value=0, max_value=12)
            balance_score = st.number_input("Balance Score (0–4)", min_value=0, max_value=4)
            gait_speed = st.number_input("Gait Speed (0–4)", min_value=0, max_value=4)
            stand_up_score = st.number_input("Stand Up Score (0–4)", min_value=0, max_value=4)
        with col4:
            qmci = st.number_input("Qmci Score (0-100)", min_value=0.0, max_value=100.0)

        submit = st.form_submit_button("Classify Games")

    if submit:
        st.session_state.inputs = {
            "name": name,
            "Age": Age,
            "care_level": care_level,
            "BMI": BMI,
            "education_level": education_level,
            "Sex": Sex,
            "mobility_type": mobility_type,
            "previous_experience": previous_experience,
            "sppb": sppb,
            "balance_score": balance_score,
            "gait_speed": gait_speed,
            "stand_up_score": stand_up_score,
            "qmci": qmci
        }
        st.session_state.show_results = True
        st.rerun()

# --- Page 2: Results ---
if st.session_state.show_results:
    try:
        inputs = st.session_state.inputs
        name = inputs["name"]

        # Encode inputs
        yes_no_map = {"Yes": 1, "No": 0}
        sex_map = {"Male": 0, "Female": 1}
        previous_experience = yes_no_map[inputs["previous_experience"]]
        mobility_type = yes_no_map[inputs["mobility_type"]]
        #hearing_impairments = yes_no_map[inputs["hearing_impairments"]]
        Sex = sex_map[inputs["Sex"]]

        # One row per game
        user_rows = []
        for game in all_games:
            user_rows.append({
                "Sex": Sex,
                "Age": inputs["Age"],
                "Previous experience": previous_experience,
                #"Hearing Problems": hearing_impairments,
                "Education Level": inputs["education_level"],
                "BMI": inputs["BMI"],
                "Care level": inputs["care_level"],
                "QMCI Points": inputs["qmci"],
                "4 m Gehtest": inputs["gait_speed"],
                "Stand-up-Test": inputs["stand_up_score"],
                "Mobility Aids": mobility_type,
                "Balance-Test": inputs["balance_score"],
                "Game": game
            })

        user_data = pd.DataFrame(user_rows)

        # Preprocess and predict
        user_data_proc = preprocessor.transform(user_data)
        predictions = model.predict(user_data_proc)
        prediction_probs = model.predict_proba(user_data_proc)
        labels = label_encoder.inverse_transform(predictions)

        # Decode to human-readable descriptions
        label_map = {
            "110": " Not able to play the game",
            "111": " Needs verbal and physical support",
            "101": " Needs physical support only",
            "011": " Needs verbal support only",
            "001": " Able to play without support"
        }

        user_data["Support Code"] = labels
        user_data["Support Description"] = user_data["Support Code"].map(label_map)
        user_data["Confidence"] = prediction_probs.max(axis=1)

        # Group games
        grouped = user_data.groupby("Support Description")["Game"].apply(list)

        st.header(f" Game Recommendations for {name}")
        for description, games in grouped.items():
            st.markdown(f"### {description}")
            st.markdown(", ".join(games))

        # Option to return
        if st.button(" Back to Form"):
            st.session_state.show_results = False
            st.rerun()

    except Exception as e:
        st.error(f" An error occurred:\n\n{e}")

