import langchain_helper as lch
import streamlit as st

st.title("Pets name generator")

animal_type = st.sidebar.selectbox("What kinda of pet do you have?", ("cat","dog","pig","rodent", "reptile"))

pet_color = st.sidebar.text_area(label="What color is your pet?",max_chars=15)

if pet_color:
    response = lch.generate_pet_name(animal_type, pet_color)
    st.text(response)