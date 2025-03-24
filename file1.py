import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# --- Set Page Config ---
st.set_page_config(page_title="Resume Ranker", page_icon="📄", layout="wide")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    #read directory from file
    for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# --- UI Layout ---
st.markdown("<h1 class='title'>🚀 AI-Powered Resume Ranking System</h1>", unsafe_allow_html=True)

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description and resumes
    documents = [job_description] + resumes
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Compute cosine similarity between job description and each resume
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Rank resumes based on similarity score
    ranked_resumes = sorted(zip(resumes, similarity_scores), key=lambda x: x[1], reverse=True)
    
    return ranked_resumes

# Streamlit UI
st.title("Resume Ranking System")


# --- Custom CSS Styling ---
st.markdown(
   """  
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stApp {
        background: url('https://images.unsplash.com/photo-1534156039819-c89418369a4f?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D') no-repeat center fixed;
        background-size: cover;
    }
    .title {
        font-size: 38px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
    }
    .upload-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }
    .rank-btn {
        display: block;
        width: 100%;
        padding: 10px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #ff5733;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: 0.3s;
    }
    .rank-btn:hover {
        background-color: #c70039;
    }
    .result-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Upload job description
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
st.subheader("📌 Upload Job Description (PDF)")
job_description_file = st.file_uploader("UPLOAD JOB DESCRIPTION (PDF)", type=["pdf"])
if job_description_file:
    job_description_text = extract_text_from_pdf(job_description_file)
    st.write("Job Description Text Extracted Successfully")
st.markdown("</div>", unsafe_allow_html=True)

# Upload multiple resumes
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
st.subheader("📂 Upload Resumes (Multiple PDFs)")

resumes_files = st.file_uploader("UPLOAD RESUMES (PDF)", type=["pdf"], accept_multiple_files=True)
resumes_texts = []
if resumes_files:
    for resume_file in resumes_files:
        resumes_texts.append(extract_text_from_pdf(resume_file))
    st.write(f"{len(resumes_files)} Resumes Uploaded Successfully")
st.markdown("</div>", unsafe_allow_html=True)



# Rank resumes when button is clicked
if st.button("Rank Resumes"):
    if job_description_text and resumes_texts:
        ranked_results = rank_resumes(job_description_text, resumes_texts)
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("📊 Ranked Resumes")
        for i, (resume_text, score) in enumerate(ranked_results):
            st.write(f"Rank {i+1}: Score {score:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Please upload both job description and resumes.")

