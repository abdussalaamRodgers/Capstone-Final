import streamlit as st
import pandas as pd
import pickle
import time

# Custom styling using Markdown and HTML
st.markdown("""
    <style>
    .title {
        font-size:40px;
        color:#ff4b4b;
        text-align:center;
        font-weight:bold;
        font-family:sans-serif;
    }
    .subtitle {
        font-size:20px;
        color:#4b7bff;
        text-align:center;
        font-style:italic;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom title
st.markdown('<p class="title">ASRacing</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Tyre Compound Prediction App</p>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.title("ASRacing Settings")

# Mapping for race names to race IDs and reverse
races = {
    "Sakhir": 0, "Jeddah": 1, "Melbourne": 2, "Imola": 3, "Miami": 4,
    "Barcelona": 5, "Monaco": 6, "Baku": 7, "Montréal": 8, "Silverstone": 9,
    "Spielberg": 10, "Le Castellet": 11, "Budapest": 12, "Spa-Francorchamps": 13,
    "Zandvoort": 14, "Monza": 15, "Marina Bay": 16, "Suzuka": 17, "Austin": 18,
    "Mexico City": 19, "São Paulo": 20, "Yas Island": 21, "Portimão": 22,
    "Montmeló": 23, "Monte-Carlo": 24, "Spa": 13, "Sochi": 26, "Istanbul": 27,
    "Al Daayen": 28, "Abu Dhabi": 30, "Spain": 31, "Mugello": 32, "Nürburg": 33,
    "Shanghai": 34, "Hockenheim": 35, "Montreal": 36, "Spielberg0": 37,
    "Spielberg1": 38, 'Silverstone2': 39, 'Silverstone3': 40, "Sakhir4": 41, 
    "Sakhir5": 42
}

reverse_race_encoding = {v: k for k, v in races.items()}  # Reverse mapping

# Compound name encoding
reverse_compound_encoding = {0: 'SOFT', 1: 'MEDIUM', 2: 'HARD', 3: 'INTERMEDIATE', 4:'WET'}

# Collecting input from the user using Sidebar
Race_Name = st.sidebar.selectbox('Race Name', list(races.keys()))
Lap = st.sidebar.number_input('Lap', min_value=0, max_value=100, value=0)
Air_Temperature = st.sidebar.number_input('Air Temperature (°C)', min_value=-30.0, max_value=50.0, value=0.0)
Humidity = st.sidebar.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=0.0)
Pressure = st.sidebar.number_input('Pressure (hPa)', min_value=500.0, max_value=2000.0, value=500.0)
Rainfall = st.sidebar.number_input('Rainfall', min_value=0, max_value=1, value=0)
Track_Temperature = st.sidebar.number_input('Track Temperature (°C)', min_value=-30.0, max_value=70.0, value=0.0)
Wind_Direction = st.sidebar.number_input('Wind Direction (°)', min_value=0, max_value=360, value=180)
Wind_Speed = st.sidebar.number_input('Wind Speed (m/s)', min_value=0.0, max_value=100.0, value=0.0)
DriverNumber = st.sidebar.number_input('Driver Number', min_value=0, max_value=31, value=2)

# Convert race name to race ID
Race = races[Race_Name]

# Load the trained model
with open('model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Input form data
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
X_train_columns = ['Race', 'Lap', 'Air Temperature', 'Humidity', 'Pressure', 'Rainfall', 
                   'Track Temperature', 'Wind Direction', 'Wind Speed', 'DriverNumber']

new_input_encoded = new_input_encoded.reindex(columns=X_train_columns, fill_value=0)

# Add a progress bar to simulate a loading state when predicting
progress_bar = st.progress(0)

# Simulate some time delay for prediction
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)

# Make prediction when the user clicks the button
if st.button('Predict Tyre Compound'):
    try:
        # Predict using the model
        predicted_compound_encoded = rf_model.predict(new_input_encoded)
        
        # Map the encoded compound prediction back to the compound name
        predicted_compound = reverse_compound_encoding[predicted_compound_encoded[0]]
        
        st.write(f"### Predicted Tyre Compound: **{predicted_compound}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add some media (image, video) for a racing feel
st.image('C:/Users/User/Desktop/Capstone Project/Images for Presentation/aif1.jfif', use_column_width=True)

# You can add more interactivity, graphs, or additional custom widgets here
st.sidebar.write("Adjust the values on the sidebar to see predictions update")

