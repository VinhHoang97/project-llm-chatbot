import streamlit as st
import requests

if st.button("Tạo không gian embedding"):
    response = requests.post("http://fastapi:8000/create-document")
    st.write("Hoàn thành")

text_input = st.text_area("Nhập câu hỏi:")

if st.button("Xử lý"):
    response = requests.post("http://fastapi:8000/process", json={"query": text_input})
    result = response.json()
    st.write(result)