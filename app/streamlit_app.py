import streamlit as st
from src.document_loader import load_pdf, split_text
from src.web_scraper import scrape_url
from src.vector_store import create_vector_store
from src.rag_chain import create_qa_chain

st.set_page_config(page_title="AI Research Assistant", layout="centered")
st.title("🧠 AI Research Assistant")

# Input source selection
option = st.radio("Choose input type:", ("PDF", "URL"))

text = None

# Handle PDF upload
if option == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        text = load_pdf(uploaded_file)

# Handle URL scraping
elif option == "URL":
    url = st.text_input("Enter a URL to scrape:")
    if url:
        text = scrape_url(url)

# Once text is available, process it
if text:
    with st.spinner("Processing and generating vector store..."):
        chunks = split_text(text)
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)

    question = st.text_input("Ask a question about the document:")

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        st.success("**Answer:**")
        st.write(answer)
