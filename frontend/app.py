import streamlit as st
import requests

if st.button("Tạo không gian embedding"):
    response = requests.post("http://fastapi:8000/create-document")
    st.write("Hoàn thành")

st.title("Hệ thống trả lời câu hỏi")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Hãy nhập câu hỏi của bạn..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        modelAnswer = requests.post("http://fastapi:8000/process", json={"query": prompt})
        response = st.write_stream(modelAnswer.json()["result"])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})