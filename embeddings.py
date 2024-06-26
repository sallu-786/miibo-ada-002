#in This file we read the text from files and create embeddings plus obtain keywords using bm25 search
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from file_handler import get_excel_text,get_csv_text
import streamlit as st
#---------------------------------------------------
import os
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
model = "kant_embed"
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")


def get_file(file):
    filename = file.name 

    if filename.endswith(('.xlsx', '.xls')):
        return get_excel_text(file)
    elif filename.endswith('.csv'):
        return get_csv_text(file)
    else: 
        raise ValueError("Unsupported file type")

def get_text_chunks(pages):  # divide text of file into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=30, 
                                          length_function=len)
    chunks = []
    for text, page_number in pages:
        for chunk in text_splitter.split_text(text):
            chunks.append({"text": chunk, "page_number": page_number})
    return chunks


#---------------------------------------------------------------------------------------------------------------
class DocumentChunk:                   #create a class to store text chunk with metadata (page number)
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
@st.cache_resource
def load_embeddings():
    #  embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
     embeddings = AzureOpenAIEmbeddings(azure_deployment=model, openai_api_version="2023-05-15")
     vector_store= FAISS.load_local("faiss_index_ai", embeddings, allow_dangerous_deserialization=True)
     return vector_store

def create_embeddings(text_chunks):
    
    # embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embeddings = AzureOpenAIEmbeddings(azure_deployment=model, openai_api_version="2023-05-15")
    documents = [DocumentChunk(page_content=chunk['text'], metadata={'page': chunk['page_number']}) 
                 for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index_ai")
    # retriever = vector_store.as_retriever(search_kwargs={'k': 1}, search_type="similarity")
    return vector_store

def bm25_search(text_chunks, query, k=1):
    docs = [DocumentChunk(page_content=chunk['text'], metadata={'page': chunk['page_number']})
             for chunk in text_chunks]
    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = scores.argsort()[-k:][::-1]
    results = [(docs[i], scores[i]) for i in top_k_indices]
    return results
