import streamlit as st

# (최소, 최대, 처음엔 어디)
length = st.slider('슬라이더', 1, 100, 50)
st.text(length)
