from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import pickle
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM to use to answer the question
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.8,
)

with open(f"dataset/doc_embeddings.pkl", "rb") as f:
    VectorStore = pickle.load(f)

retriever = VectorStore.as_retriever(
    search_type="similarity", search_kwargs={"k": 15})

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)


def query(input):
    input += " (answer based on context, else answer I don't know)"
    response = qa(input)
    return response["result"]


def fallback(input):
    response = qa(input + "(answer question based on context provided, else answer I don't know)")
    return response["result"]


def preprocess(input):
    input += "(answer question based on context provided, else answer I don't know)"
    response = qa.run(input).strip()
    return response == "I don't know."
