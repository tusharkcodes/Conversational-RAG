# 1_create_vector_db_faiss.py
import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # Swapped from Chroma

PDF_FOLDER = "books"
FAISS_DB_PATH = "faiss_selfhelp_db"  # New folder for FAISS index
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

docs = []
for pdf_file in Path(PDF_FOLDER).glob("*.pdf"):
    print(f"Loading {pdf_file.name}...")
    loader = PyMuPDFLoader(str(pdf_file))
    docs.extend(loader.load())

print(f"\nTotal pages: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Add book title to metadata
for chunk in chunks:
    chunk.metadata["book"] = Path(chunk.metadata["source"]).stem.replace("_", " ")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("\nCreating FAISS index... (this takes 1-3 minutes on M3)")
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save persistently (FAISS saves to file)
vectorstore.save_local(FAISS_DB_PATH)

print(f"\nSUCCESS! FAISS DB saved in folder: {FAISS_DB_PATH}")
print("You can now close this script. Run it again only if you add new books.")