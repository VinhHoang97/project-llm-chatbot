from fastapi import FastAPI
from pydantic import BaseModel
import logging

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

class InputDataFormat(BaseModel):
    text: str
    
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    

@app.post("/create-document")
def create_embedding():
    file_path='formatData/data.json'
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[].content",
        text_content=False)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(all_splits, hf_embeddings, persist_directory="./chroma_langchain_db")
    vectorstore.persist()
    
    print("test vectorstore", vectorstore)

@app.post("/process")
def process(query: str):
    chroma_store = Chroma(
        embedding_function=hf_embeddings, 
        persist_directory="./chroma_langchain_db"
    )
    query = "tin tức nổi bật nhất hôm nay?"
    results = chroma_store.similarity_search(query)
    return {"result": results[0]}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)