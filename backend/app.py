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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n"])
    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf_embeddings, persist_directory=persist_directory)

def format_docs(docs):
    print(type(docs))
    print("Tét docs: ", docs)
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


    template = """Bạn là một trợ lý hữu ích giúp trả lời câu hỏi dựa trên tài liệu. 
    Tài liệu: {context}
    Câu hỏi: {question}
    Nếu không có tài liệu nào trong {context} thì trả lời "Bạn không biết"
    Trả lời theo định dạng sau:
    ## Câu trả lời: ...
    ## Tài liệu: {context}
    """

    # Tạo PromptTemplate cho hệ thống hỏi đáp
    prompt_template = ChatPromptTemplate.from_template(
        template,
    )

    # Gọi retrieval_chain_rag_fusion để lấy tài liệu
    retrieval_output = retrieval_chain_rag_fusion.invoke({"question": query})

    # lisSourceURL = [doc.metadata.get("sourceURL", "") ]
    print(type(retrieval_output))
    listDocument = format_docs(retrieval_output)
    
    # print("Retrieval source URL: ", lisSourceURL)

    # Sử dụng RetrievalQA chain
    qa_chain = (
        {"context": itemgetter("listDocument"), 
        "question": itemgetter("question")} 
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    # Gọi response từ query
    response = qa_chain.invoke({"question":query, "context": listDocument})

    return {
        "result": response,
        # "sourceURL": lisSourceURL,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)