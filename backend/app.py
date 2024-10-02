from fastapi import FastAPI
from pydantic import BaseModel
import logging

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class InputDataFormat(BaseModel):
    query: str
    
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    

@app.post("/create-document")
def create_embedding():
    MODEL_NAME = "keepitreal/vietnamese-sbert"
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    persist_directory = 'db'
    
    file_path='formatData/data.json'
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[].content",
        text_content=False)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n"])
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf_embeddings, persist_directory=persist_directory)
    vectorstore.persist()

@app.post("/process")
def process(input: InputDataFormat):
    MODEL_NAME = "keepitreal/vietnamese-sbert"
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    persist_directory = 'db'
    chroma_store = Chroma(persist_directory=persist_directory, embedding_function=hf_embeddings)
    # Tạo retriever từ Chroma
    retriever = chroma_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    query = input.model_dump()["query"]
        
    template = """Dưới đây là các tài lệu liên quan đến câu hỏi của bạn:
        TÀI LỆU: {document}
        Trả lời câu hỏi sau: {question}
        
        Hướng dẫn:
        - Câu trả lời đầy đủ và chi tiết.
        - Không tự tạo đáp án nếu không thể trả lời
        - Không sử dụng các cụm từ dẫn đến một văn bản khác như "theo tài liệu, theo đường dẫn, theo thông tin,..." và các cụm từ tương tự.
        - Đường dẫn của tài liệu chứa câu trả lời
        - Trả lời theo format:
        # Câu trả lời:
        ...
        # Tham khảo:
        ... 
    #     """
    
    prompt_template = PromptTemplate(
        template=template, input_variables=["document", "question"]
    )
    
    # Khởi tạo ChatOllama
    llm = ChatOllama(
        temperature=0,
        base_url="http://ollama:11434/",
        model="llama3.2:1b",
        streaming=True,
        top_k=10,  # Độ đa dạng câu trả lời
        top_p=0.3,  # Mức độ tập trung của văn bản sinh ra
        num_ctx=3072,  # Kích thước cửa sổ ngữ cảnh
    )

    # Sử dụng RunnableSequence để kết hợp prompt và llm
    llm_chain = RetrievalQA(
        llm=llm,
        retriever=retriever,
        prompt=prompt_template,
        return_source_documents=True
    )

    # Gọi response từ query
    response = llm_chain({"query": query})

    # Trả kết quả
    documents = "\n".join([doc.page_content for doc in response["source_documents"]])
    return {
        "result": response["result"],
        "documents": documents
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)