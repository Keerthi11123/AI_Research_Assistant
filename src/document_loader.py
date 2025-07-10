import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import requests
from io import BytesIO
from bs4 import BeautifulSoup

def load_pdf(file_like):
    """
    Load PDF text from a file-like object (e.g., Streamlit upload or BytesIO)
    """
    doc = fitz.open(stream=file_like.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_pdf_from_url(url):
    """
    Download and load text from a PDF URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_stream = BytesIO(response.content)
        return load_pdf(pdf_stream)
    except Exception as e:
        return f"❌ Error loading PDF from URL: {e}"

def load_text_from_webpage(url):
    """
    Scrape and return visible text from a non-PDF webpage
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        return f"❌ Error fetching URL: {e}"

def load_document(input_type, input_value):
    """
    Unified interface to load content based on input type:
    - input_type: "pdf" or "url"
    - input_value: Streamlit file uploader object or URL string
    """
    if input_type == "pdf":
        return load_pdf(input_value)
    elif input_type == "url":
        # Try to handle PDF URLs separately
        if input_value.lower().endswith(".pdf"):
            return load_pdf_from_url(input_value)
        else:
            return load_text_from_webpage(input_value)
    else:
        return "❌ Invalid input type."

def split_text(text, chunk_size=400, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.create_documents([text])
    return chunks