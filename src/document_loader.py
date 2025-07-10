import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def load_pdf(path):
    doc = fitz.open(path)
    full_text = "\n".join([page.get_text() for page in doc])
    # Try removing everything after 'References'
    if "references" in full_text.lower():
        full_text = full_text.lower().split("references")[0]
    return full_text


def split_text(text, chunk_size=400, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.create_documents([text])
    return chunks

