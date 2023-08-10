import pickle
import streamlit as st
#Example data

data={"name":"Linda", "age": 25, "city":"New York"}

#Save data to a file using pickle
f=open("data.pkl","wb")
pickle.dump(data,f)
f.close()

#Load data from the file
f=open("data.pkl","rb")
d=pickle.load(f)

#Print the loaded data
st.write(d)