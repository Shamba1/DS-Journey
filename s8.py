import streamlit as st
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Streamlit app title
st.title("Diabetes Prediction")

# Read the dataset
data = pd.read_csv(r"C:\Users\Thando\Downloads\Diabetes (1).csv")

# Display initial data exploration
st.write("First few rows of the dataset:")
st.write(data.head())
st.write("Shape of the dataset:")
st.write(data.shape)
st.write("Summary statistics:")
st.write(data.describe())
st.write("Correlation matrix:")
st.write(data.corr())

# Prepare data
x = data.iloc[:, 0:8]
y = data.iloc[:, -1]

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and fit the Logistic Regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# Save the trained classifier using pickle
with open("classifier.pkl", "wb") as h:
    pickle.dump(lr, h)

# Streamlit UI for prediction
st.sidebar.header("User Input Features")
pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 800, 1)
bp = st.sidebar.slider("Blood Pressure", 0, 200, 1)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 1)
insulin = st.sidebar.slider("Insulin", 30, 200, 30)
bmi = st.sidebar.slider("BMI", 20, 80, 20)
DBF = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 4.0, 0.1)
AGE = st.sidebar.slider("Age", 0, 120, 1)

# Create a data array for prediction
user_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, DBF, AGE]])

# Load the trained classifier using pickle
with open("classifier.pkl", "rb") as h:
    loaded_lr = pickle.load(h)

# Make prediction
prediction = loaded_lr.predict(user_data)

# Display prediction
st.write("Prediction:", prediction)

# Display classification report heatmap
if st.checkbox("Show Classification Report Heatmap"):
    y_pred = loaded_lr.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("Classification Report Heatmap:")
    sns.heatmap(report_df, annot=True)
