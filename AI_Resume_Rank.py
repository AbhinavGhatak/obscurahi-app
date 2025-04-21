import os, tempfile
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def parse_resume(fp):
    text = ''
    try:
        with open(fp, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                text += p.extract_text() or ''
    except Exception as e:
        print(f"Error reading {fp}: {e}")
    return text

def preprocess_and_vectorize(resume_texts, job_desc):
    documents = resume_texts + [job_desc]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix

def keyword_count_norm(text, keywords):
    words = text.lower().split()
    count = sum(words.count(k.lower()) for k in keywords)
    return count / len(words) if words else 0

def evaluate_resumes(paths, job_desc, keywords):
    # Extract text from each resume
    texts = [parse_resume(p) for p in paths]
    # Vectorize resumes + job description
    tfidf = preprocess_and_vectorize(texts, job_desc)
    # Separate vectors
    jd_vec = tfidf[-1]
    res_vecs = tfidf[:-1]
    # Cosine similarity
    sims = cosine_similarity(res_vecs, jd_vec).flatten()
    # Build scores list
    scores = []
    for i, txt in enumerate(texts):
        kscore = keyword_count_norm(txt, keywords)
        total = sims[i] * 0.7 + kscore * 0.3
        scores.append((i, total * 100))
    # Sort descending
    return sorted(scores, key=lambda x: x[1], reverse=True)
