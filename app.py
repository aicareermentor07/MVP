import streamlit as st
import os
import pdfplumber
import docx
import requests
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()
st.set_page_config(page_title="AI Job Finder", page_icon="💼", layout="wide")

# API Keys (set these in your .env or Streamlit secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# HELPERS
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        # Keep messages user-friendly
        st.warning("PDF extraction had an issue. If your resume is a scanned image, OCR is required.")
    return text

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception:
        st.warning("DOCX extraction had an issue. Please try another file version.")
        return ""

def analyze_resume_with_gpt(resume_text):
    prompt = f"""
    Extract the following from this resume:
    1. Total years of experience
    2. List of technical skills (normalized)
    3. Up to 3 key projects with one-line summaries
    4. Suggested job titles this person should target

    Return a concise plain-text response that a machine can parse (you may use labels like 'Suggested job titles:' and 'List of technical skills:').

    Resume:
    {resume_text}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an expert career coach."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()

def parse_analysis_to_candidates(analysis_text):
    """
    Extract Suggested Job Titles and some skills from GPT analysis for querying Adzuna.
    Returns a cleaned list of query strings.
    """
    candidates = []
    lines = [l.strip() for l in analysis_text.splitlines() if l.strip()]

    # Look for explicit "Suggested job titles" or similar patterns
    for i, line in enumerate(lines):
        low = line.lower()
        if "suggested job titles" in low or line.lower().startswith("suggested job titles"):
            parts = line.split(":", 1)
            if len(parts) > 1:
                titles = [t.strip() for t in re.split(r'[,\|;/]+', parts[1]) if t.strip()]
                candidates.extend(titles)
            # Also capture following short lines as possible titles
            j = i + 1
            while j < len(lines) and len(candidates) < 10:
                nxt = lines[j]
                if len(nxt.split()) <= 8:
                    more = [t.strip() for t in re.split(r'[,\|;/]+', nxt) if t.strip()]
                    candidates.extend(more)
                    j += 1
                else:
                    break
            break

    # Also look for a skills line to add skill-based queries
    for i, line in enumerate(lines):
        low = line.lower()
        if "list of technical skills" in low or line.lower().startswith("skills:") or "technical skills" in low:
            parts = line.split(":", 1)
            right = parts[1] if len(parts) > 1 else ""
            skills = [s.strip() for s in re.split(r'[,\|;/]+', right) if s.strip()]
            candidates.extend(skills[:6])
            # check next lines for additional skill bullets
            j = i + 1
            while j < len(lines) and len(candidates) < 12:
                nxt = lines[j]
                if "," in nxt or len(nxt.split()) <= 10:
                    more = [s.strip() for s in re.split(r'[,\|;/]+', nxt) if s.strip()]
                    candidates.extend(more)
                    j += 1
                else:
                    break
            break

    # Fallback: use first few words of analysis as a query
    if not candidates:
        first = " ".join(analysis_text.split()[:6]).strip()
        if first:
            candidates = [first]

    # Final cleanup: unique, reasonable length
    cleaned = []
    for c in candidates:
        if 2 <= len(c) <= 80 and c.lower() not in [x.lower() for x in cleaned]:
            cleaned.append(c)
    return cleaned

def fetch_real_jobs(candidates, results=20):
    """
    candidates: list or string (we coerce to list)
    Tries candidate queries and multiple locations; returns list of job dicts.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        st.error("ADZUNA_APP_ID/ADZUNA_APP_KEY missing. Set them in your environment or Streamlit secrets.")
        return []

    # coerce to list
    if isinstance(candidates, str):
        candidates = [candidates]
    elif not isinstance(candidates, list):
        candidates = list(candidates)

    collected = []
    seen = set()
    per_page = min(50, results)

    # Locations to try (narrow -> broad)
    locations = ["Chennai", "Bangalore", "Hyderabad", "Coimbatore", "Madurai", "Tamil Nadu", "India", ""]

    # Priority role keywords to attempt
    priority_roles = [
        "Software Development & Engineering",
        "Engineering Operations",
        "Machine Learning",
        "Data Analytics & Business Intelligence"
    ]

    # Queries to attempt: start with parsed candidates, then priority roles, then common fallbacks
    queries = [q for q in candidates if q] + priority_roles + ["Data Engineer", "Software Engineer", "Developer", "Backend Developer", "Data Analyst"]

    for q in queries:
        if len(collected) >= results:
            break
        for loc in locations:
            if len(collected) >= results:
                break

            url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
            params = {
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "results_per_page": per_page,
                "what": q,
                "where": loc,
            }

            try:
                resp = requests.get(url, params=params, timeout=12)
            except Exception:
                # silently continue; no UI debug prints
                continue

            if resp.status_code != 200:
                # handle auth errors explicitly for user
                if resp.status_code in (401, 403):
                    st.error("Adzuna authentication error — check ADZUNA_APP_ID and ADZUNA_APP_KEY.")
                    return []
                continue

            try:
                data = resp.json().get("results", [])
            except Exception:
                continue

            for job in data:
                jid = job.get("id") or (job.get("title","") + "|" + job.get("company",{}).get("display_name",""))
                if jid in seen:
                    continue
                seen.add(jid)
                collected.append({
                    "title": job.get("title"),
                    "company": job.get("company", {}).get("display_name", ""),
                    "location": job.get("location", {}).get("display_name", ""),
                    "description": (job.get("description") or "")[:300].replace("\n", " ").strip(),
                    "redirect_url": job.get("redirect_url") or job.get("url") or ""
                })
                if len(collected) >= results:
                    break

            # small sleep to respect rate limits
            time.sleep(0.25)

    return collected[:results]

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("💼 AI Job Finder (India)")
st.write("Upload your resume → Get top **20 latest jobs** matching your skills, projects, and experience.")

uploaded_file = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    # Extract resume text
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text.strip():
            st.warning("PDF text extraction returned no text. If your resume is a scanned image, OCR is required (not supported in this MVP). Please upload a text PDF or DOCX.")
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    if not resume_text.strip():
        st.error("Could not extract text from the uploaded file. Try a different resume file (text PDF or DOCX).")
    else:
        st.success("✅ Resume uploaded successfully!")

        # Analyze with GPT
        with st.spinner("Analyzing resume with AI..."):
            analysis = analyze_resume_with_gpt(resume_text)

        st.subheader("📝 Profile Extracted from Resume")
        st.text(analysis)

        # Parse candidates (job titles/skills)
        candidates = parse_analysis_to_candidates(analysis)
        # Show detected candidate queries (non-debug, concise)
        if candidates:
            st.write("Detected candidate queries:", ", ".join(candidates))
        else:
            st.write("Detected candidate queries: (using fallback queries)")

        with st.spinner("Fetching latest jobs..."):
            jobs = fetch_real_jobs(candidates, results=20)

        st.subheader("💼 Top 20 Job Matches")
        st.info("This is beta. Auto-apply will be added soon.")

        if not jobs:
            st.warning("⚠️ No jobs found for your profile. Try broadening resume keywords or location.")
        else:
            # Build table rows
            rows = []
            for j in jobs:
                apply_link = j.get("redirect_url") or ""
                apply_md = f"[Apply]({apply_link})" if apply_link else "N/A"
                rows.append({
                    "Role": j.get("title") or "N/A",
                    "Company": j.get("company") or "N/A",
                    "Location": j.get("location") or "N/A",
                    "Job description": j.get("description") or "N/A",
                    "Apply now link": apply_md
                })

            df = pd.DataFrame(rows)
            # Render as markdown table so links are clickable
            md_table = df.to_markdown(index=False)
            st.markdown(md_table, unsafe_allow_html=True)
