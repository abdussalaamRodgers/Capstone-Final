import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

#title
st.title('ASRacing')

#subtitle
st.subheader('Tyre Compound Prediction App')

# Collecting input from the user
Race = st.number_input('Race', min_value=0, max_value=100)
Lap = st.number_input('Lap', min_value=0, max_value=100)
Air_Temperature = st.number_input('Air Temperature (°C)', min_value=-30.0, max_value=50.0)
Humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0)
Pressure = st.number_input('Pressure (hPa)', min_value=500.0, max_value=2000.0)
Rainfall = st.number_input('Rainfall', min_value=0.0, max_value=1.0)
Track_Temperature = st.number_input('Track Temperature (°C)', min_value=-30.0, max_value=70.0)
Wind_Direction = st.number_input('Wind Direction (°)', min_value=0, max_value=360)
Wind_Speed = st.number_input('Wind Speed (m/s)', min_value=0.0, max_value=100.0)
DriverNumber = st.number_input('Driver Number', min_value=1, max_value=99)

# Create new input data as a DataFrame
new_data = {
    'Race': [Race],
    'Lap': [Lap],
    'Air Temperature': [Air_Temperature],
    'Humidity': [Humidity],
    'Pressure': [Pressure],
    'Rainfall': [Rainfall],
    'Track Temperature': [Track_Temperature],
    'Wind Direction': [Wind_Direction],
    'Wind Speed': [Wind_Speed],
    'DriverNumber': [DriverNumber]
}

new_input = pd.DataFrame(new_data)

# One-hot encode the new input data
new_input_encoded = pd.get_dummies(new_input)

# Reindex to ensure the new input matches the model's training data
# Assuming you saved X_train columns when training the model:
X_train_columns = ['Race', 'Lap', 'Air Temperature', 'Humidity', 'Pressure', 'Rainfall', 
                   'Track Temperature', 'Wind Direction', 'Wind Speed', 'DriverNumber']

new_input_encoded = new_input_encoded.reindex(columns=X_train_columns, fill_value=0)

# Make prediction when the user clicks the button
if st.button('Predict Tyre Compound'):
    try:
        predicted_compound = rf_model.predict(new_input_encoded)
        st.write(f"Predicted Compound: {predicted_compound[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

