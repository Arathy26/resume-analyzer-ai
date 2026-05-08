# ============================================
# AI RESUME ANALYZER - FULL RAG VERSION
# LangChain + Groq + ChromaDB + Streamlit
# ============================================

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import PyPDF2
import docx

# --- LOAD API KEY ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- SET UP AI MODEL ---
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

# --- READ PDF ---
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- READ WORD DOC ---
def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# --- CHUNK TEXT ---
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# --- CREATE VECTOR DB ---
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    return vector_db

# --- SEARCH CHUNKS ---
def search_chunks(vector_db, question, k=3):
    results = vector_db.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

# --- ANALYZE RESUME ---
def analyze_resume(vector_db, chunks, job_role):
    # get all chunks as context
    context = "\n\n".join(chunks[:10])
    
    prompt = PromptTemplate(
        input_variables=["context", "job_role"],
        template="""
        You are an expert HR recruiter and career coach.
        Analyze this resume for the job role: {job_role}
        
        Resume Content:
        {context}
        
        Give a detailed analysis in this exact format:
        
        🎯 OVERALL SCORE: (give a score out of 100)
        
        ✅ STRENGTHS:
        (list 3-5 strong points of this resume)
        
        ⚠️ WEAKNESSES:
        (list 3-5 weak points or missing things)
        
        💡 IMPROVEMENTS:
        (list 5 specific actionable improvements)
        
        🔑 MISSING KEYWORDS:
        (list important keywords missing for {job_role} role)
        
        📝 SUMMARY:
        (2-3 sentence overall assessment)
        
        🏆 HIRING RECOMMENDATION:
        (Would you hire this person? Yes/No/Maybe - explain why)
        """
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "job_role": job_role
    }).content
    return result

# --- ANSWER QUESTION ---
def answer_question(vector_db, question):
    context = search_chunks(vector_db, question)
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an expert career coach.
        Use this resume content to answer the question.
        
        Resume Content:
        {context}
        
        Question: {question}
        
        Give a clear helpful answer:
        """
    )
    
    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "question": question
    }).content
    return result

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📋",
    layout="centered"
)

st.title("📋 AI Resume Analyzer")
st.subheader("Upload your resume and get AI-powered feedback!")
st.divider()

# --- FILE UPLOAD ---
st.markdown("### 📂 Upload Your Resume")
input_method = st.radio(
    "Choose file type:",
    ["📄 PDF Resume", "📝 Word Resume", "✍️ Paste Resume Text"]
)

extracted_text = ""

if input_method == "📄 PDF Resume":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Reading resume..."):
            extracted_text = read_pdf(uploaded_file)
        st.success(f"✅ Resume loaded! ({len(extracted_text)} characters)")

elif input_method == "📝 Word Resume":
    uploaded_file = st.file_uploader("Upload Word Doc", type=["docx"])
    if uploaded_file:
        with st.spinner("Reading resume..."):
            extracted_text = read_docx(uploaded_file)
        st.success(f"✅ Resume loaded! ({len(extracted_text)} characters)")

elif input_method == "✍️ Paste Resume Text":
    extracted_text = st.text_area(
        "Paste your resume here",
        height=300,
        placeholder="Paste your full resume text here..."
    )

# --- JOB ROLE INPUT ---
if extracted_text:
    st.divider()
    st.markdown("### 🎯 Target Job Role")
    job_role = st.text_input(
        "What job role are you applying for?",
        placeholder="e.g. AI Engineer, Data Scientist, Software Developer, CA..."
    )

    st.divider()

    # --- RAG PIPELINE ---
    st.markdown("### 🔄 RAG Pipeline")

    with st.spinner("⚙️ Step 1: Chunking resume..."):
        chunks = chunk_text(extracted_text)
    st.success(f"✅ Step 1: Created {len(chunks)} chunks")

    with st.spinner("⚙️ Step 2: Creating embeddings + vector DB..."):
        vector_db = create_vector_db(chunks)
    st.success("✅ Step 2: Vector database ready!")

    st.divider()

    # --- OPTION 1 - FULL ANALYSIS ---
    st.markdown("### 🔍 Option 1 — Full Resume Analysis")
    
    if st.button("🚀 Analyze My Resume", use_container_width=True):
        if not job_role.strip():
            st.warning("⚠️ Please enter a job role first!")
        else:
            with st.spinner("🤖 AI is analyzing your resume..."):
                result = analyze_resume(vector_db, chunks, job_role)

            st.success("✅ Analysis Complete!")
            st.divider()
            st.markdown("### 📊 Resume Analysis Report")
            st.markdown(result)

            st.download_button(
                label="⬇️ Download Analysis Report",
                data=result,
                file_name="resume_analysis.txt",
                mime="text/plain"
            )

    st.divider()

    # --- OPTION 2 - ASK QUESTIONS ---
    st.markdown("### 💬 Option 2 — Ask About Your Resume")
    question = st.text_input(
        "Ask anything about your resume:",
        placeholder="e.g. What skills am I missing? How can I improve my summary?"
    )

    if st.button("💬 Get Answer", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Please type a question!")
        else:
            with st.spinner("🔍 Searching resume + generating answer..."):
                result = answer_question(vector_db, question)

            st.success("✅ Answer Ready!")
            st.markdown("### 💡 Answer")
            st.markdown(result)

            with st.expander("🔍 See resume sections used"):
                context = search_chunks(vector_db, question)
                st.write(context)

st.divider()
st.caption("Built with LangChain + Groq + ChromaDB + HuggingFace 🚀")
# .\venv\Scripts\Activate.ps1( activate virtual environment command for Windows PowerShell)
# streamlit run app.py( command to run the Streamlit app)
