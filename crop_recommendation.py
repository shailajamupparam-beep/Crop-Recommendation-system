import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("archive (1).zip")

X = df[["N", "P","K","temperature", "humidity","ph","rainfall"]]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.title("Smart Crop Recommendation System")
st.write("Enter the soil and environmental conditions " "to get a suitable crop recommendation.")
st.sidebar.header("Model Performance")
st.sidebar.write("Accuracy:",round(accuracy * 100, 2),"%")

N = st.number_input("Nitrogen (N)",min_value=0.0,max_value=200.0,value=50.0)
P = st.number_input("Phosphorus (P)",min_value=0.0,max_value=200.0,value=50.0)
K = st.number_input("Potassium (K)",min_value=0.0,max_value=250.0,value=50.0)

temperature = st.number_input("Temperature (°C)",min_value=0.0,max_value=50.0,value=25.0)
humidity = st.number_input("Humidity (%)",min_value=0.0,max_value=100.0,value=60.0)

ph = st.number_input("Soil pH",min_value=0.0,max_value=14.0,value=6.5)

rainfall = st.number_input("Rainfall (mm)",min_value=0.0,max_value=500.0,value=100.0)

if st.button("Recommend Crop"):
    input_data = [[N,P,K,temperature,humidity,ph,rainfall]]
    prediction = model.predict(input_data)
    crop = label_encoder.inverse_transform(prediction)
    st.success(f"Recommended Crop: {crop[0].upper()}")
