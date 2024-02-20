import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings

#codes for preparing context
pdf_reader = PdfReader("dataset/Strata-Management-Act-757-English.pdf")
text = ""
for page in pdf_reader.pages:
  text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)

chunks = text_splitter.split_text(text=text)
embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-mpnet-base-v2")
VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
VectorStore.similarity_search_with_score()
with open(f"doc_embeddings_AI.pkl", "wb") as f:
      pickle.dump(VectorStore, f)
print("embeddings created")