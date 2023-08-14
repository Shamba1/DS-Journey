import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

st.title("Diabetes Prediction")
data = pd.read_csv(r"C:\Users\Thando\Downloads\Diabetes (1).csv")
st.write(data.head())
st.write(data.shape)
st.write(data.describe())
st.write(data.corr())

x=data.iloc[:,0:8]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
st.write("Predicted Values:")
st.write(y_pred)
st.write("Actual Values:")
st.write(y_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cf=confusion_matrix(y_test,y_pred)
st.write(cf)
st.write("Classification Report for Testing Dataset:")
#st.write(sns.heatmap(classification_report(y_test,y_pred),annot=True))
report=classification_report(y_test,y_pred)
st.write(report)

h=open("classifier.pkl","wb")
pickle.dump(lr,h)
h.close() 