
# Libraries imported
import streamlit as st
import numpy as np
import joblib

# Loading trained model, scaler, and encoder 
knn_model = joblib.load("knn_lung_cancer_model.joblib")
scaler = joblib.load("lung_cancer_scaler.joblib")
gender_encoder = joblib.load("gender_encoder.joblib")

# lable is set to display the end result 
label_map = {0: "No Lung Cancer Detected", 1: "Lung Cancer Detected"}

# Main UI page starts form here 
st.title("Lung Cancer Risk Prediction")
st.write("This tool uses a KNN model trained on survey data to estimate lung cancer risk."
         "It is a learning project and not a medical diagnosis.")

st.header("Basic Details")
gender = st.selectbox("Gender", ["MALE", "FEMALE"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.header("Symptoms")
st.write("Select Yes or No for each question.")

# Creates a reusable Yes/No question using two checkboxes.
def yes_no(label, key_prefix):
    st.markdown(f"**{label}**")
    # Creates unique keys for the Yes and No checkboxes
    # so Streamlit can keep track of each question separately.
    yes_key = f"{key_prefix}_yes"
    no_key = f"{key_prefix}_no"

    # If the user selects "Yes", automatically uncheck "No".
    def on_yes_change():
        if st.session_state[yes_key]:
            st.session_state[no_key] = False
    # If the user selects "No", automatically uncheck "Yes".
    def on_no_change():
        if st.session_state[no_key]:
            st.session_state[yes_key] = False

    # Creates two columns so the Yes and No checkboxes can appear side by side.
    col1, col2 = st.columns(2)
    with col1:  # yes checkbox display
        yes = st.checkbox("Yes", key=yes_key, on_change=on_yes_change)
    with col2:  # No checkbox display
        no = st.checkbox("No", key=no_key, on_change=on_no_change)
    return yes

# Questions asked to patient 
smoking = yes_no("Do you smoke?", "smoking")
yellow_fingers = yes_no("Do you have yellow fingers?", "yellow_fingers")
anxiety = yes_no("Do you experience anxiety?", "anxiety")
peer_pressure = yes_no("Do you experience peer pressure?", "peer_pressure")
chronic_disease = yes_no("Do you have any chronic disease?", "chronic_disease")
fatigue = yes_no("Do you experience fatigue?", "fatigue")
allergy = yes_no("Do you have any allergies?", "allergy")
wheezing = yes_no("Do you experience wheezing?", "wheezing")
alcohol_consuming = yes_no("Do you consume alcohol?", "alcohol_consuming")
coughing = yes_no("Do you experience coughing?", "coughing")
shortness_of_breath = yes_no("Do you experience shortness of breath?", "shortness_of_breath")
swallowing_difficulty = yes_no("Do you have difficulty swallowing?", "swallowing_difficulty")
chest_pain = yes_no("Do you experience chest pain?", "chest_pain")

# Prediction based on the symptoms 
if st.button("Predict"):

    if gender in gender_encoder.classes_:
        gender_encoded = gender_encoder.transform([gender])[0]
    else:
        gender_encoded = np.mean(gender_encoder.transform(gender_encoder.classes_))

    # Important thing here is the order must exactly match x.columns from the notebook:
    # ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
    #  'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING',
    #  'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
    input_data = np.array([[
        gender_encoded,
        age,
        int(smoking),
        int(yellow_fingers),
        int(anxiety),
        int(peer_pressure),
        int(chronic_disease),
        int(fatigue),
        int(allergy),
        int(wheezing),
        int(alcohol_consuming),
        int(coughing),
        int(shortness_of_breath),
        int(swallowing_difficulty),
        int(chest_pain)
    ]])

    # Scale the user's input using the same scaler,that was used during model training.
    input_scaled = scaler.transform(input_data)
    # Predict whether lung cancer is detected or not using the trained KNN model.
    prediction = knn_model.predict(input_scaled)
    # Converts the 0 and 1 into msg as mentioned above
    result = label_map[prediction[0]]

    # Displays result to user in UI page
    if prediction[0] == 1:
        st.error(result + " — please consult a doctor.")
    else:
        st.success(result)
