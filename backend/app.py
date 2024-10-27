from fastapi import FastAPI
from pydantic import BaseModel
import logging
from operator import itemgetter
from langchain.load import dumps, loads
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import requests


class InputDataFormat(BaseModel):
    query: str
    
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "keepitreal/vietnamese-sbert"
hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

print("Loading Done: ",hf_embeddings)

persist_directory = 'db'
vectorstore = None


@app.post("/import-url")
def import_url(input: dict):
   #call local host 3000 to import new url
    url = input["urlName"]
    response = requests.get("http://host.docker.internal:3000/data?urlName="+url)
    return {"message": "Imported data from URL: " + url}

@app.post("/get-question")
def get_question(input: InputDataFormat):
    llm = ChatOllama(
        temperature=0,
        base_url="http://ollama:11434/",
        model="llama3.2:1b",
        streaming=True,
        top_k=5,  # Độ đa dạng câu trả lời
        top_p=0.3,  # Mức độ tập trung của văn bản sinh ra
        num_ctx=3072,  # Kích thước cửa sổ ngữ cảnh
    )
    query = input.model_dump()["query"]

    template = """
    Câu trả lời trước đó của tôi là: {question}.
    Giúp tôi tạo một câu hỏi ngẫu nhiên để cập nhật tin tức mới nhất dựa trên câu trả lời trước đó.
    """
    # Tạo PromptTemplate cho hệ thống hỏi đáp
    prompt_template = ChatPromptTemplate.from_template(
        template,
    )
    qa_chain = (
        {"question": itemgetter("question")} 
        | prompt_template
        | llm
        | StrOutputParser()
    )
    # Gọi response từ query
    response = qa_chain.invoke({"question": query})

    return {
        "result": response,
    }
    
# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["sourceURL"] = record.get("sourceURL")
    metadata["keywords"] = record.get("keywords")

    return metadata
@app.post("/create-document")
def create_embedding():
    global vectorstore
    
    file_path='formatData/data.json'
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        metadata_func=metadata_func,
        content_key="content",)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100, separators=["\n\n", "\n", '.'])
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf_embeddings, persist_directory=persist_directory)

@app.post("/process")
def process(input: InputDataFormat):
    print("Loading Chroma...: ",hf_embeddings)
    retriever = vectorstore.as_retriever()

    query = input.model_dump()["query"]

    # Khởi tạo LLM ChatOllama
    llm = ChatOllama(
        temperature=0,
        base_url="http://ollama:11434/",
        model="llama3.2:1b",
        streaming=True,
        top_k=5,  # Độ đa dạng câu trả lời
        top_p=0.3,  # Mức độ tập trung của văn bản sinh ra
        num_ctx=3072,  # Kích thước cửa sổ ngữ cảnh
    )
    
    # Tạo retriever từ Chroma
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
                Generate multiple search queries related to: {question} \n
                Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_rag_fusion 
            | llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
    )
    retrieval_chain_rag_fusion = generate_queries | retriever.map()
    reranked_results = retrieval_chain_rag_fusion.invoke({"question": query})
    
    fused_scores = {}
    k=60
    for docs in reranked_results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            # print('\n')
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

        # final reranked result
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
    template = """Bạn làm một trợ lý trả lời câu hổi dựa vào tài liệu liên quan, nếu không có tài liệu thì trả lời là "tôi không biết" và không tạo các chi tiết không có trong document :
    
    Tài liệu: {context}
    Câu hỏi: {question}
    """

    # Tạo PromptTemplate cho hệ thống hỏi đáp
    prompt_template = ChatPromptTemplate.from_template(
        template,
    )

    # Sử dụng RetrievalQA chain
    qa_chain = (
        {"context": itemgetter("document"), 
        "question": itemgetter("question")} 
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Gọi response từ query
    response = qa_chain.invoke({"question":query, "document":str(list(reranked_results[0])[0].page_content)})

    return {
        "result": response,
        "document": reranked_results[0]
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)