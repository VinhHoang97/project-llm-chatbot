from fastapi import FastAPI
from pydantic import BaseModel
import logging

from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


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
    query = "tin tức nổi bật nhất hôm nay?"
    results = chroma_store.similarity_search(query)
    return {"result": results[0].page_content}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)