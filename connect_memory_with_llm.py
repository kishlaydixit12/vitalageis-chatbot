import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
)

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ----------------------------
# STEP 1: Load LLM
# ----------------------------
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )
    return llm

# ----------------------------
# STEP 2: Load All Documents
# ----------------------------
def load_all_documents(data_folder):
    all_docs = []
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
        elif filename.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {filename}")
            continue
        docs = loader.load()
        all_docs.extend(docs)
    print(f"✅ Loaded {len(all_docs)} documents total")
    return all_docs

# ----------------------------
# STEP 3: Create / Update FAISS
# ----------------------------
def get_or_create_vectorstore(data_folder, db_path, embedding_model):
    if os.path.exists(db_path):
        print("🧠 Loading existing FAISS database...")
        db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

        # Check for new data
        print("🔍 Checking for new documents to add...")
        all_docs = load_all_documents(data_folder)
        if all_docs:
            db.add_documents(all_docs)
            db.save_local(db_path)
            print("✅ FAISS database updated with new data!")
    else:
        print("🆕 Creating new FAISS database from data folder...")
        all_docs = load_all_documents(data_folder)
        db = FAISS.from_documents(all_docs, embedding_model)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.save_local(db_path)
        print("✅ New FAISS database created and saved!")
    return db

# ----------------------------
# STEP 4: Custom Prompt
# ----------------------------
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say you don't know — don't make up an answer.
Only provide information from the given context.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# ----------------------------
# STEP 5: Live Folder Watcher
# ----------------------------
class DataFolderWatcher(FileSystemEventHandler):
    def __init__(self, data_folder, db_path, embedding_model):
        self.data_folder = data_folder
        self.db_path = db_path
        self.embedding_model = embedding_model

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".pdf", ".txt", ".csv", ".docx")):
            print(f"\n📄 New file detected: {event.src_path}")
            db = FAISS.load_local(self.db_path, self.embedding_model, allow_dangerous_deserialization=True)
            loader = None
            if event.src_path.endswith(".pdf"):
                loader = PyPDFLoader(event.src_path)
            elif event.src_path.endswith(".txt"):
                loader = TextLoader(event.src_path)
            elif event.src_path.endswith(".csv"):
                loader = CSVLoader(event.src_path)
            elif event.src_path.endswith(".docx"):
                loader = Docx2txtLoader(event.src_path)

            if loader:
                new_docs = loader.load()
                db.add_documents(new_docs)
                db.save_local(self.db_path)
                print("✅ New file added to FAISS database!\n")

def start_folder_watcher(data_folder, db_path, embedding_model):
    event_handler = DataFolderWatcher(data_folder, db_path, embedding_model)
    observer = Observer()
    observer.schedule(event_handler, path=data_folder, recursive=False)
    observer.start()
    print("👀 Watching for new files in data folder... (PDF/TXT/CSV/DOCX)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ----------------------------
# STEP 6: Build Retrieval QA
# ----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = get_or_create_vectorstore(DATA_FOLDER, DB_FAISS_PATH, embedding_model)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 10}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Start watching folder in background
threading.Thread(target=start_folder_watcher, args=(DATA_FOLDER, DB_FAISS_PATH, embedding_model), daemon=True).start()

# ----------------------------
# STEP 7: User Query
# ----------------------------
while True:
    user_query = input("\n💬 Write Query Here (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        print("👋 Exiting chatbot...")
        break

    response = qa_chain.invoke({'query': user_query})

    print("\n======================")
    print("🧾 RESULT:")
    print(response["result"])
    print("\n📚 SOURCE DOCUMENTS:")
    for doc in response["source_documents"]:
        print("•", doc.metadata.get("source", "Unknown"))
    print("======================\n")
