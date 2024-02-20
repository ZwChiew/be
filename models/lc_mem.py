from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

with open(f"dataset/doc_embeddings.pkl", "rb") as f:
    VectorStore = pickle.load(f)

retriever = VectorStore.as_retriever(
    search_type="similarity", search_kwargs={"k": 15})

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1)

qa = RetrievalQA.from_chain_type(
    llm=model, chain_type="stuff", retriever=retriever, return_source_documents=False)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=2,
    return_messages=True
)

tools = [
    Tool(
        name="Knowledge Base",
        func=qa.run,
        description=(
            "Answer based on context provided"
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=model,
    max_iterations=5,
    verbose=False,
    early_stopping_method='generate',
    handle_parsing_errors=True,
    memory=conversational_memory)


def query(question, answer, mes):
    conversational_memory.save_context({"input": question}, {"output": answer})
    try:
        response = agent.run(mes)
        return response
    except Exception as e:
        print("using fallback")
        response = qa.run(mes)
        return response