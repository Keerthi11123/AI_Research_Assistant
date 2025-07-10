import argparse
from src.document_loader import load_pdf, split_text
from src.web_scraper import scrape_url
from src.vector_store import create_vector_store, load_vector_store
from src.rag_chain import create_qa_chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Path to PDF")
    parser.add_argument("--url", help="URL to scrape")
    parser.add_argument("--ask", help="Question to ask")
    args = parser.parse_args()

    print(f"📄 Loading PDF from {args.pdf}" if args.pdf else f"🌐 Scraping content from {args.url}")

    if args.pdf:
        raw_text = load_pdf(args.pdf)
    elif args.url:
        raw_text = scrape_url(args.url)
    else:
        raise ValueError("Provide --pdf or --url")

    print("✂️ Splitting text into chunks...")
    chunks = split_text(raw_text, chunk_size=400, overlap=50)

    print("🧠 Creating vector store...")
    vector_store = create_vector_store(chunks)

    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(args.ask)
    print("\n🧩 Top retrieved chunk:\n")
    print(docs[0].page_content[:500])  # show first 500 characters

    print("🤖 Building QA chain...")
    qa_chain = create_qa_chain(vector_store)

    print(f"❓ Question: {args.ask}")
    answer = qa_chain.invoke({"query": args.ask})
    print("✅ Answer:", answer['result'])


if __name__ == "__main__":
    main()
