import streamlit as st
import os
import pdfplumber
import docx
import pandas as pd
from difflib import SequenceMatcher
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Resume Fixer + Job Matcher", page_icon="🚀")

# Load API key from Streamlit Secrets
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def get_ai_resume_feedback(resume_text, target_role="Software Engineer"):
    prompt = f"""
    You are an ATS resume expert.
    Resume Text: {resume_text}

    Task:
    1. Give an ATS Match Score (0-100) for the target role: {target_role}.
    2. List 5 most important missing keywords.
    3. Suggest 3 improvements to make it recruiter-friendly.
    4. Rewrite 3 work experience bullet points using impact + keywords.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",   # You can also use "gpt-4" if needed
        messages=[
            {"role": "system", "content": "You are a professional resume coach and ATS expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=600,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def match_score(resume, jd):
    return SequenceMatcher(None, resume.lower(), jd.lower()).ratio()

def get_job_matches(resume_text, job_csv="jobs.csv"):
    try:
        jobs = pd.read_csv(job_csv)
    except Exception as e:
        st.error(f"⚠️ Could not load jobs.csv: {e}")
        return pd.DataFrame(columns=["title", "company", "description", "score"])

    if jobs.empty:
        st.warning("⚠️ Job list is empty. Please add data to jobs.csv.")
        return pd.DataFrame(columns=["title", "company", "description", "score"])

    jobs["score"] = jobs["description"].apply(lambda x: match_score(resume_text, x))
    top_jobs = jobs.sort_values("score", ascending=False).head(5)
    return top_jobs[["title", "company", "description", "score"]]

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🚀 Resume Fixer + Job Matcher MVP")
st.write("Upload your resume and instantly get AI feedback + top job matches!")

# Upload file
uploaded_file = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])
target_role = st.text_input("🎯 Target Role", value="Software Engineer")

if uploaded_file:
    # Extract resume text
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    st.success("✅ Resume uploaded successfully!")

    # AI Resume Fixer
    with st.spinner("Analyzing resume with AI..."):
        ai_feedback = get_ai_resume_feedback(resume_text, target_role)

    st.subheader("📊 Resume Feedback (AI-powered)")
    st.write(ai_feedback)

    # Job Matching
    with st.spinner("Finding top job matches..."):
        top_jobs = get_job_matches(resume_text)

    st.subheader("💼 Top Job Matches")
    st.info("Note: This is only for testing purposes, not a real job")
    for _, row in top_jobs.iterrows():
        st.markdown(f"""
        **{row['title']}** at *{row['company']}*  
        📌 {row['description']}  
        🔥 Match Score: **{round(row['score']*100, 1)}%**  
        """)

    # Download improved resume (future step)
    st.info("📥 Note : Download improved resume (PDF/DOCX) will be provided in future")
