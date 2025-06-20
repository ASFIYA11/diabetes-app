#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    return pd.read_csv(url)

df = load_data()

# Train Model
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient data below to predict diabetes risk.")

# Input Form
with st.form("user_input"):
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose", 0, 200, 120)
    bp = st.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 10, 100, 30)
    submitted = st.form_submit_button("Predict")

    if submitted:
        user_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(user_data)[0]
        label = "Diabetic" if prediction == 1 else "Not Diabetic"
        st.subheader(f"Prediction: {label}")

# Visualizations
st.subheader("📊 Glucose Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['Glucose'], kde=True, bins=30, ax=ax1)
plt.axvline(glucose, color='red', linestyle='--', label='Your Input')
plt.legend()
st.pyplot(fig1)

st.subheader("📊 BMI Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['BMI'], kde=True, bins=30, ax=ax2)
plt.axvline(bmi, color='red', linestyle='--', label='Your Input')
plt.legend()
st.pyplot(fig2)

st.subheader("🧮 Model Performance (on training data)")
st.text(classification_report(y, model.predict(X)))


# In[ ]:




