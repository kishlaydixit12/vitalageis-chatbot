import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

# -----------------------------
# CONFIG
# -----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

# -----------------------------
# FUNCTION: Load or Reload Vectorstore
# -----------------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def reload_vectorstore():
    """Manually reload FAISS DB when new data is added"""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    st.session_state['vectorstore'] = db
    st.success("✅ Data reloaded successfully!")

# -----------------------------
# PROMPT SETUP
# -----------------------------
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.title("🧠 VITAL AEGIS Chatbot")

    # -----------------------------
    # Data Reload Button
    # -----------------------------
    if st.button("🔄 Reload Data"):
        with st.spinner("Reloading FAISS data..."):
            reload_vectorstore()

    # Load vectorstore into session (only once)
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = get_vectorstore()

    vectorstore = st.session_state['vectorstore']

    # -----------------------------
    # Chat Interface
    # -----------------------------
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("💬 Ask Vital Aegis anything!")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't make up an answer.
        Don't provide anything outside of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_docs = response["source_documents"]

            result_to_show = result + "\n\n📚 **Source Docs:**\n" + str(source_docs)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

# -----------------------------
if __name__ == "__main__":
    main()
