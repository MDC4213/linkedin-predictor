import streamlit as st

st.title("My app")
st.header("Hello World!")

st.write("My application")

if st.button("Click me"):
    st.balloons()
    st.write("Clicked!")

name = st.text_input("Name: ", "")
if name:
    st.write(f"My name is {name}")

    