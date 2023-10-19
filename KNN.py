import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# Load the dataset
df = pd.read_csv(r"heart_data.csv")
del df['id']
df.index = df['index']
df['age'] = df['age'].apply(lambda x: int(x/365))
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
features = df[features]

label = df['cardio'].to_numpy

scaler = MinMaxScaler()
Xtransformed = scaler.fit_transform(features)
Xtrain = Xtransformed[:30000, :]
Xval = Xtransformed[30000:50000, :]
Xtest = Xtransformed[50000:, :]

label = df['cardio'].to_numpy()
trainLabel = label[:30000]
valLabel = label[30000:50000]
testLabel = label[50000:]

k = np.arange(1, 21, 1)
train_score = []
for i in k:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(Xtrain, trainLabel)
    train_score.append(model.score(Xtrain, trainLabel))

model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(Xtrain, trainLabel)

# Create a Streamlit app
st.title("Heart Disease Prediction App")
st.header("Enter Feature Values")

# User input for feature values
age = st.slider("Age", 20, 100, 40)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 0, 300, 160)
weight = st.number_input("Weight (kg)", 0, 200, 70)
ap_hi = st.number_input("Systolic Blood Pressure", 0, 300, 120)
ap_lo = st.number_input("Diastolic Blood Pressure", 0, 200, 80)
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
gluc = st.selectbox("Glucose Level", [1, 2, 3])
smoke = st.selectbox("Smoking", [0, 1])
alco = st.selectbox("Alcohol Intake", [0, 1])
active = st.selectbox("Physical Activity", [0, 1])

# Map gender to numeric values
gender = 0 if gender == "Male" else 1

# Preprocess the user input
user_input = [age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]
user_input = np.array(user_input).reshape(1, -1)

# Predict the user input
user_prediction = model.predict(user_input)
result = "Positive" if user_prediction == 1 else "Negative"

# Display prediction result
st.subheader("Prediction Result:")
st.write(f"The prediction for heart disease is {result}.")
