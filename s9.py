import streamlit as st

# Page title
st.title("Digital Regenesys Course Profile")

# Course data
course_data = [
    ("Data Science", "15 Weeks", "Python, R, Tableau"),
    ("Cyber Security", "6 Weeks", "Basic, Advance"),
    ("Digital Marketing", "6 Weeks", "Basic, Advance")
]

# Table to display course information
st.write("Digital Regenesys offers:")
st.table(course_data)
