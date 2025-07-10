"""
Model Selector:
This pipeline uses HuggingFace models for local inference without any API keys.
You can switch between models like:

- google/flan-t5-small (fast, limited reasoning)
- google/flan-t5-base (balanced performance)
- google/flan-t5-large (better reasoning and full-paragraph answers)

Simply change the model name in the `pipeline()` function below.
"""
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


def create_qa_chain(vector_store):
    # Load a stronger local model (e.g., 'flan-t5-small', 'flan-t5-base', or 'flan-t5-large' can be used based on your requirement)
    pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    retriever = vector_store.as_retriever()
    qa_chain =RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain