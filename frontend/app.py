import streamlit as st
import requests

if st.button("Xử lý"):
    response = requests.post("http://fastapi:8000/process", json={"text": "hello"})
    result = response.json()
    st.write(result)