from base import *
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

store = Chroma(collection_name="notes", embedding_function=embeddings, persist_directory="./chroma_db")

@tool
def search_notes(query: str) -> str:
  """"""
