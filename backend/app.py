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

@app.post("/import-url")
def import_url(input: {urlName: str}):
   #call local host 3000 to import new url
    response = requests.post("http://localhost:3000/data?urlName={}".format(input["urlName"]))
    return {"message": "Imported URL"}
    

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n"])
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf_embeddings, persist_directory=persist_directory)
    vectorstore.persist()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

@app.post("/process")
def process(input: InputDataFormat):
    MODEL_NAME = "keepitreal/vietnamese-sbert"
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    persist_directory = 'db'
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=hf_embeddings)
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
    template = """Bạn là trợ lý hữu ích tạo ra nhiều truy vấn tìm kiếm dựa trên một truy vấn đầu vào duy nhất. \nTạo nhiều truy vấn tìm kiếm liên quan đến: {question} \n
    Đầu ra (4 câu truy vấn):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_rag_fusion 
            | llm
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
    )
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion


    template = """Dưới đây là các tài liệu liên quan đến câu hỏi của bạn:
    TÀI LIỆU: {context}

    Câu hỏi: {question}

    Hướng dẫn cách trả lời câu hỏi:
    - Câu trả lời đầy đủ và chi tiết.
    - Không tự tạo đáp án nếu không thể trả lời
    - Không sử dụng các cụm từ dẫn đến một văn bản khác như "theo tài liệu, theo đường dẫn, theo thông tin,..." và các cụm từ tương tự.

    Trả lời theo format:
    # Câu trả lời:
    ...
    # Tham khảo:
    ... 
    """

    # Tạo PromptTemplate cho hệ thống hỏi đáp
    prompt_template = ChatPromptTemplate.from_template(
        template,
    )

    # Sử dụng RetrievalQA chain
    qa_chain = (
        {"context": retrieval_chain_rag_fusion, 
        "question": itemgetter("question")} 
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Gọi response từ query
    response = qa_chain.invoke({"question":query})

    return {
        "result": response,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)