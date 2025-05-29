#import os
# from langchain_groq import ChatGroq
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceBgeEmbeddings

# def initialize_llm():
#     return ChatGroq(
#         temperature=0,
#         groq_api_key="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",
#         model_name="llama-3.3-70b-versatile"  # or any other model supported
#     )

# def create_or_load_vector_db(pdf_path, db_path):
#     embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
#     if not os.path.exists(db_path):
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load()
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         texts = splitter.split_documents(docs)
#         db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
#         db.persist()
#     else:
#         db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
#     return db
# llm_setup.py

import os
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

def initialize_llm():
    print("Env keys:", list(os.environ.keys()))
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("❌ Missing GROQ_API_KEY in environment variables or Hugging Face secrets.")
    
    # ✅ Supported model: "llama3-70b-8192", "mixtral-8x7b-32768", etc.
    return ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
#alternative "llama3-70b-8192" if above doesnot work

def create_or_load_vector_db(pdf_path, db_path):
    # Use a real huggingface model that works with sentence embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    if not os.path.exists(db_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
        db.persist()
    else:
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    return db
