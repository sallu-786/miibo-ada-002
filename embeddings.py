
#in This file we read the text from files and create embeddings plus obtain keywords using bm25 search
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from file_handler import get_pdf_text,get_text,get_word_text,get_ppt_text,get_excel_text,get_csv_text
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv()
model = "kant_embed"
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")


def get_file(files):
    if isinstance(files, list):
        text_chunks = []
        for file in files:
            filename = file.name 
            if filename.endswith('.pdf'):
                text_chunks.extend(get_pdf_text(file))
            elif filename.endswith('.txt'):
                text_chunks.extend(get_text(file))
            elif filename.endswith(('.docx', '.doc')):
                text_chunks.extend(get_word_text(file))
            elif filename.endswith(('.pptx', '.ppt')):
                text_chunks.extend(get_ppt_text(file))
            elif filename.endswith(('.xlsx', '.xls')):
                text_chunks.extend(get_excel_text(file))
            elif filename.endswith('.csv'):
                text_chunks.extend(get_csv_text(file))
            else: 
                raise ValueError(f"Unsupported file type: {filename}")
        
        return text_chunks
    else:
        filename = files.name 
        if filename.endswith('.pdf'):
            return get_pdf_text(files)
        elif filename.endswith('.txt'):
            return get_text(files)
        elif filename.endswith(('.docx', '.doc')):
            return get_word_text(files)
        elif filename.endswith(('.pptx', '.ppt')):
            return get_ppt_text(files)
        elif filename.endswith(('.xlsx', '.xls')):
            return get_excel_text(files)
        elif filename.endswith('.csv'):
            return get_csv_text(files)
        else: 
            raise ValueError(f"Unsupported file type: {filename}")


def get_text_chunks(pages):  # divide text of file into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, 
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

def ensemble_retriever(text_chunks, query):
  
    docs = [DocumentChunk(page_content=chunk['text'], metadata={'page': chunk['page_number']})
                for chunk in text_chunks]
    bm25_retriever = BM25Retriever.from_documents(docs,search_kwargs={"k": 3})
    embeddings = AzureOpenAIEmbeddings(azure_deployment=model, openai_api_version="2023-05-15")
    documents = [DocumentChunk(page_content=chunk['text'], metadata={'page': chunk['page_number']}) 
                 for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.25, 0.75])
    final_result = ensemble_retriever.invoke(query)
    return final_result