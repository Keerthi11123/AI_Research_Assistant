from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_hf_embedding_model():
    # Create once and reuse (avoids reloading model multiple times)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(documents, persist_directory="embeddings"):
    embedding = get_hf_embedding_model()
    vectordb = Chroma.from_documents(documents, embedding, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def load_vector_store(persist_directory="embeddings"):
    embedding = get_hf_embedding_model()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb
