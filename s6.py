import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
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

# Make predictions
y_pred = lr.predict(x_test)

# Display predicted and actual values
st.write("Predicted Values:")
st.write(y_pred)
st.write("Actual Values:")
st.write(y_test)

# Calculate and display confusion matrix
cf = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(cf)

# Generate classification report and display as heatmap
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.write("Classification Report for Testing Dataset:")
st.write(report_df)

# Display heatmap using seaborn
st.write("Classification Report Heatmap:")
sns.heatmap(report_df, annot=True)

# Save the trained classifier using joblib
with open("classifier.pkl", "wb") as h:
    pickle.dump(lr, h)
