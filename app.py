import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain Ecosystem
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. ENVIRONMENT & SECURITY ---
def load_security_environment():
    """Loads environment variables securely."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        st.error("🚨 Critical: GROQ_API_KEY missing in .env")
        st.stop()

# --- 2. AI ENGINE INITIALIZATION ---
@st.cache_resource
def initialize_ai_engine():
    """Boots up the LLM, Embeddings, and Vector Store."""
    # Ultra-fast LLM via Groq
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1, # Low temperature for medical precision
        max_tokens=1024
    )
    
    # Local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Local persistent vector database
    vectorstore = Chroma(
        collection_name="medical_guidelines",
        embedding_function=embeddings,
        persist_directory="./chroma_db_medical"
    )
    
    return llm, embeddings, vectorstore

# --- 3. INGESTION PIPELINE ---
def process_document(uploaded_file, vectorstore):
    """Processes uploaded medical PDFs into ChromaDB."""
    with st.spinner("Ingesting medical guidelines..."):
        # Handle file temporarily for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        
        # Chunking strategy for medical texts (preserves paragraph context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        splits = text_splitter.split_documents(docs)
        
        vectorstore.add_documents(splits)
        os.unlink(tmp_file_path) # Clean up temp file
        st.sidebar.success(f"Indexed {len(splits)} chunks into ChromaDB.")

# --- 4. RAG ORCHESTRATION & HYBRID FALLBACK ---
def build_hybrid_rag_chain(llm, vectorstore):
    """Constructs the LCEL chain with the Hybrid Fallback Guardrail."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # The Hybrid Fallback Prompt
    system_template = """You are an elite clinical guidelines copilot. Follow these strict execution rules:
    
    1. CONTEXT EVALUATION: Analyze the provided 'Medical Guidelines' to answer the 'User Query'.
    2. PRIMARY PROTOCOL (RAG HIT): If the answer exists within the guidelines, provide a precise answer and append [Source: Internal Clinical Guidelines].
    3. HYBRID FALLBACK (RAG MISS): If the guidelines DO NOT contain the answer, you MUST state exactly: "The uploaded guidelines do not cover this topic. Relying on general medical knowledge:" followed by your answer.
    4. GUARDRAIL: Never hallucinate document citations. Always end general knowledge answers with "Disclaimer: Consult a licensed healthcare professional."
    
    Medical Guidelines:
    {context}
    
    User Query: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(system_template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # LCEL Pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- 5. FRONTEND / UI ---
def build_ui():
    """Constructs the Streamlit interface."""
    st.set_page_config(page_title="Medical Guidelines Copilot", page_icon="⚕️", layout="wide")
    st.title("⚕️ Medical Guidelines Copilot")
    st.caption("Powered by Llama 3.3, ChromaDB, & HuggingFace")

    # Initialize Backend
    load_security_environment()
    llm, embeddings, vectorstore = initialize_ai_engine()
    rag_chain = build_hybrid_rag_chain(llm, vectorstore)

    # Sidebar: Knowledge Base Management
    st.sidebar.header("Knowledge Base Setup")
    uploaded_file = st.sidebar.file_uploader("Upload Medical Guidelines (PDF)", type="pdf")
    
    if uploaded_file and st.sidebar.button("Index Document"):
        process_document(uploaded_file, vectorstore)

    # Chat State Management
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "System online. Awaiting query..."}
        ]

    # Render History
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Enter clinical query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving clinical context..."):
                # 1. Manually retrieve docs to capture metadata for the UI
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                source_docs = retriever.invoke(prompt)
                
                # 2. Generate the AI response
                response = rag_chain.invoke(prompt)
                st.write(response)
                
                # 3. The Hackathon Winning Feature: Expandable Citations
                if source_docs:
                    with st.expander("📄 View Source Guidelines & Page Numbers"):
                        for i, doc in enumerate(source_docs):
                            # PyPDFLoader automatically extracts the page number into metadata!
                            page_num = doc.metadata.get('page', 0) + 1 if 'page' in doc.metadata else 'Unknown'
                            st.markdown(f"**Reference {i+1} | Page {page_num}**")
                            # Show a snippet of the exact text it read
                            st.info(doc.page_content[:400] + "...")
                
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- EXECUTION ---
if __name__ == "__main__":
    build_ui()