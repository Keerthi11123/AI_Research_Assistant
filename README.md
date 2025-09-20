# 🧠 AI Research Assistant (LangChain + HuggingFace)

This project is a **local, privacy-first AI Research Assistant** that lets you upload a PDF or scrape a webpage and then **ask questions about it using a Retrieval-Augmented Generation (RAG) pipeline**.

Powered by **LangChain**, **Hugging Face models**, and **local embeddings**, it works without requiring OpenAI or Hugging Face API tokens.

---

## 🚀 Features

- ✅ Ask questions about uploaded PDFs or scraped URLs
- ✅ Built-in document chunking and semantic search
- ✅ Fully local, no API keys required
- ✅ Fast inference using `flan-t5-base` or other HuggingFace models
- ✅ Optional Streamlit UI

---

## ⚙️ How It Works

> This project uses a **Retrieval-Augmented Generation (RAG)** pipeline, built with:
- 🧠 **HuggingFace Transformers** for local text generation
- 📚 **SentenceTransformers** for embedding and similarity search
- 🔎 **ChromaDB** as a local vector store
- 🔗 **LangChain** for chaining it all together
- 🎛️ **Streamlit** for an optional web UI

---

## 🔁 Local Model Flexibility

Choose from a range of open-source HuggingFace models based on your need for speed or reasoning:

| Model             | Size   | Speed 🐇🐢   | Answer Quality 🧠           |
|------------------|--------|-------------|-----------------------------|
| `flan-t5-small`  | ~77M   | 🐇 Fastest   | ⚠️ Basic/Short              |
| `flan-t5-base`   | ~250M  | ✅ Balanced  | ✅ Solid reasoning          |
| `flan-t5-large`  | ~800M  | ⚠️ Slow      | 🧠 Detailed output          |
| `flan-t5-xl`     | ~3B    | 🐢 Slower    | 🧠 Deep, contextual answer  |


🧩 *Switch models easily in* `src/rag_chain.py`:
```python
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)

To switch models:
1. Open `src/rag_chain.py`
2. Modify the `pipeline()` line to use the model you want:
```python
pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)

---

## Setup

1. Clone the repo:
```bash
git clone https://github.com/Keerthi11123/AI_Research_Assistant.git
cd AI_Research_Assistant
