from fastapi import FastAPI
from pydantic import BaseModel
import logging

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain import PromptTemplate, LLMChain


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf_embeddings, persist_directory=persist_directory)
    vectorstore.persist()

@app.post("/process")
def process(input: InputDataFormat):
    MODEL_NAME = "keepitreal/vietnamese-sbert"
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    persist_directory = 'db'
    chroma_store = Chroma(persist_directory=persist_directory, embedding_function=hf_embeddings)

    query = input.model_dump()["query"]
    document = chroma_store.similarity_search(query)
    
    ollama = Ollama(base_url="http://localhost:11434", model="llama2")
    
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
        """
    
    prompt_template = PromptTemplate(
        template=template, input_variables=["document", "question"]
    )
    
    llm_chain = LLMChain(llm=ollama, prompt=prompt_template)
    response = llm_chain.run({"document": document, "question": query})

    return {"result": response}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)