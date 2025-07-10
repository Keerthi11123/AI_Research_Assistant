import streamlit as st
from src.document_loader import load_document, split_text
from src.vector_store import create_vector_store
from src.rag_chain import create_qa_chain

# Page setup
st.set_page_config(page_title="AI Research Assistant", layout="centered")
st.title("🧠 AI Research Assistant")

# Session state init
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "content_loaded" not in st.session_state:
    st.session_state.content_loaded = False
if "last_input" not in st.session_state:
    st.session_state.last_input = None

# Manual reset button
if st.button("🔄 Reset App"):
    st.session_state.qa_chain = None
    st.session_state.content_loaded = False
    st.session_state.last_input = None
    st.rerun()

# Step 1: Choose input type
input_type = st.radio("Choose input type:", ("PDF", "URL"))
text = None
new_input = None

# Step 2: Handle PDF upload or URL input
if input_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        new_input = uploaded_file.name
        if st.session_state.last_input != new_input:
            st.session_state.qa_chain = None
            st.session_state.content_loaded = False
            st.session_state.last_input = new_input

        if not st.session_state.content_loaded:
            with st.spinner("Loading PDF..."):
                text = load_document("pdf", uploaded_file)
                chunks = split_text(text)
                vector_store = create_vector_store(chunks)
                st.session_state.qa_chain = create_qa_chain(vector_store)
                st.session_state.content_loaded = True
            st.success("✅ PDF loaded and processed!")

elif input_type == "URL":
    url = st.text_input("Enter a URL (PDF or webpage):")
    if url:
        new_input = url.strip()
        if st.session_state.last_input != new_input:
            st.session_state.qa_chain = None
            st.session_state.content_loaded = False
            st.session_state.last_input = new_input

        if not st.session_state.content_loaded:
            with st.spinner("Loading URL..."):
                text = load_document("url", url)
                chunks = split_text(text)
                vector_store = create_vector_store(chunks)
                st.session_state.qa_chain = create_qa_chain(vector_store)
                st.session_state.content_loaded = True
            st.success("✅ URL loaded and processed!")

# Step 3: Show question input when ready
if st.session_state.qa_chain:
    question = st.text_input("💬 Ask a question about the content:")
    if question.strip():
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(question)
        st.success("✅ Answer:")
        st.markdown(f"> {answer}")
