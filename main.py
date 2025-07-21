from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

def load_documents(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            documents.extend(loader.load())
    return documents

def prepare_knowledge_base(documents):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore

def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="mistral")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa
