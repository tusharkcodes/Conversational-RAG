from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

FAISS_DB_PATH = "faiss_selfhelp_db"

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_faiss(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        
        for s in splits:
            s.metadata["file_id"] = file_id
        
        if os.path.exists(FAISS_DB_PATH):
            db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(splits)
        else:
            db = FAISS.from_documents(splits, embeddings)

        db.save_local(FAISS_DB_PATH)
        return True
    
    except Exception as e:
        print("Error:", e)
        return False

def delete_doc_from_faiss(file_id: int):
    try:
        db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = db.docstore._dict.values()

        remaining = [d for d in docs if d.metadata.get("file_id") != file_id]

        new_db = FAISS.from_documents(remaining, embeddings)
        new_db.save_local(FAISS_DB_PATH)

        print("Rebuilt FAISS index without file:", file_id)
        return True
    except Exception as e:
        print("Error:", e)
        return False
