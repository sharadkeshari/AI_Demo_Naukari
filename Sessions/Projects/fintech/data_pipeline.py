from dotenv import load_dotenv
load_dotenv()

import json
import hashlib
from datetime import datetime
from pathlib import Path
#from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

PDF_FOLDER = './bajaj_pdfs'
CHROMA_DIR = './bajajbot_chroma_db'
COLLECTION_NAME = 'bajaj_policy'
LOG_FILE ="./processed_files.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0


def get_chunk_hash(text):
    return hashlib.md5(text.strip().encode()).hexdigest()


def load_log():
    try:
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
def save_log(log:dict):
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f)


def load_and_chunk(filepath):
    pdfreader = PdfReader(filepath)
    docs = [
    Document(
        page_content=page.extract_text() or "",
        metadata={"page": i}
    )
    for i, page in enumerate(pdfreader.pages)
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata['source'] = Path(filepath).name 
        chunk.metadata["ingested_on"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chunk.metadata["chunk_hash"] = get_chunk_hash(chunk.page_content)

    return chunks


def get_existing_hashes(vectorstore:Chroma):
    existing = vectorstore.get(include=["metadatas"])
    hashes = set()
    for metadata in existing["metadatas"]:
        if metadata and "chunk_hash" in metadata:
            hashes.add(metadata["chunk_hash"])

    return hashes

# HNSW (Hierarchical Navigable Small World), TF-IDF (Term frequency--Inverse doucment frequency), 
def run_pipeline():
# Chroma uses dense vector similarity search, not hybrid search.
# Recent versions of Chroma support hybrid search APIs and RRF-based ranking when configured with sparse vector indexes.
    # Path(PDF_FOLDER).mkdir(exist_ok=True)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(collection_name=COLLECTION_NAME, 
                         embedding_function=embedding_model,
                           persist_directory=CHROMA_DIR)
    
    print("Chroma DB loaded:",vectorstore._collection.count())
    existing_hashes = get_existing_hashes(vectorstore)
    log = load_log()
    print("Existing hashes:", len(existing_hashes))

    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    print("PDF files found:", pdf_files)

    if not pdf_files:
        print("  No PDFs found. Drop PDFs into the folder and run again.")
        return
    for pdf_path in pdf_files:
        chunks = load_and_chunk(pdf_path)
        for chunk in chunks:
            if chunk.metadata["chunk_hash"] not in existing_hashes:
                print("I am inside hash for new chunks")
                vectorstore.add_documents([chunk])
                existing_hashes.add(chunk.metadata["chunk_hash"])
            
            log[pdf_path.name] ={
                "last_processed_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_chunks': len(chunks),

            }

    save_log(log)
    print("Chroma DB updated:",vectorstore._collection.count())



run_pipeline()










text= "emi is 3500 per month"
text_1= "emi is 5000 per month"
print(get_chunk_hash(text))
print(get_chunk_hash(text_1))