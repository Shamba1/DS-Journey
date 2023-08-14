import streamlit as st

#Title

st.title("Streamlit Example")


#Subheader
st.subheader("Subheader Example")

#Text
st.text("This is some G shit")

#Markdown
st.markdown("### This is a markdown heading")
st.markdown("## This is a markdown heading")
st.markdown("#### This is a markdown heading")

#Displaying data

data=[1,2,3,4,5]
st.write("Data",data)

#Dataframe

import pandas as pd

df=pd.DataFrame({"col1":[1.1,2.2,3.3], "col2": [4.4,5.5,6.6] })

st.write("DataFrame:", df)

#Plotting 

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,10,100)

y=np.sin(x)

plt.plot(x,y)

st.pyplot(plt)

#Interactive Widgets

name=st.text_input("Enter your name")

st.write("Hello,",name)

age=st.slider("Select your age:",0,100,25)

st.write("Age:",age)


color=st.selectbox("Select a color:", ["Red","Yellow","Green"])

st.write("Color:",color)

#Button


if st.button("Click me"):
    st.write("Button clicked")

#Checkbox

checkbox=st.checkbox("Check me")

if checkbox:

    st.write("Checkbox checked my bruh!")

#Sidebar

st.sidebar.title("Sidebar")

st.sidebar.text("This is a sidebar")

age=st.number_input("Enter your age:", min_value=0, max_value=100, value=25)
st.write("Age:",age)

#Date Imput
selected_date=st.date_input("Select a date:")
st.write("Selected date:", selected_date)

#Imput time

selected_time=st.time_input("Select a time:")
st.write("Selected time:", selected_time)

#Color

st.color_picker("choose your fav. color")

#File Upload

st.file_uploader('upload a file')