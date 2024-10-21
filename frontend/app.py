import streamlit as st
import requests


# questions = [
#     'tin tức hôm nay',
#     'Vụ lỡ đồi cướp mạng sống tại Hà Giang như thế nào?',
#     'Thiệt hại vụ lỡ đồi cướp mạng sống tại Hà Giang như thế nào?',
#     'Thời tiết hôm nay',
# ]

if url := st.text_input("Nhập URL"):
    # response = requests.post("http://fastapi:8000/import-url", json={"urlName": url})
    # st.write(response.json()["message"])
    st.write("Crawl dữ liệu từ URL thành công") 

if st.button("Tạo không gian embedding"):
    response = requests.post("http://fastapi:8000/create-document")
    st.write("Hoàn thành")

st.title("Hệ thống trả lời câu hỏi")

if st.button("Gợi ý câu hỏi!"):
    message = "Tin tức hôm nay"  # Mặc định câu hỏi nếu không có dữ liệu trước đó

    # Kiểm tra nếu danh sách st.session_state.messages không trống
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]  # Lấy phần tử cuối cùng mà không xóa
        message = last_message.get("content", message)  # Lấy nội dung hoặc dùng mặc định
    
    # Gửi request với câu hỏi
    response = requests.post("http://fastapi:8000/get-question", json={"query": message})
    st.write(response.json()["result"])
    

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
        response = st.write(modelAnswer.json()["result"])
        st.h1("Link nguồn:")
        st.write(modelAnswer.json()["sourceURL"])
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": modelAnswer.json()["result"]})