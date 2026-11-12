"""Hybrid similarity ranker: semantic embeddings + school-ecosystem signals + role fit."""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIGNALS = {
    "School leadership": r"principal|head ?master|head ?mistress|school director|trustee|chairman|superintendent|school owner|founder",
    "Advisory / consulting": r"consultant|advisor|adviser|mentor|coach|trainer|accreditation|curriculum|school improvement",
    "Multi-school / network": r"schools|school network|group of schools|chain|federation|trust|franchise|multiple schools",
    "K-12 context": r"k-?12|cbse|icse|\bib\b|igcse|cambridge|primary|secondary|school",
    "Education entrepreneurship": r"edtech|ed-tech|education startup|founder|co-?founder|entrepreneur",
}
ROLE_FIT = r"principal|director|trustee|owner|founder|consultant|advisor|adviser|head"
W_SEM, W_SIG, W_ROLE = 0.55, 0.30, 0.15

_model = None
def _embed(texts):
    """Sentence embeddings if available, else TF-IDF vectors (documented fallback)."""
    global _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = _model or SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(_model.encode(texts, normalize_embeddings=True)), "MiniLM embeddings"
    except Exception:
        m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
        return m.toarray(), "TF-IDF (fallback)"

def profile_text(r):
    return " . ".join(str(r.get(c, "") or "") for c in ["headline", "role", "organization", "summary", "experience"])

def signals(text):
    t = text.lower()
    return {k: bool(re.search(p, t)) for k, p in SIGNALS.items()}

def rank(df, ref_text, exclude_url=None, top_n=25):
    df = df.copy()
    if exclude_url:
        df = df[df["url"] != exclude_url]
    texts = [profile_text(r) for _, r in df.iterrows()]
    vecs, method = _embed([ref_text] + texts)
    sem = np.clip(cosine_similarity(vecs[:1], vecs[1:])[0], 0, 1)
    ref_sig = signals(ref_text)
    wanted = [k for k, v in ref_sig.items() if v] or list(SIGNALS)
    rows = []
    for (_, r), s, txt in zip(df.iterrows(), sem, texts):
        sg = signals(txt)
        hit = [k for k in wanted if sg[k]]
        sig_score = len(hit) / len(wanted)
        role_score = 1.0 if re.search(ROLE_FIT, f"{r.get('headline','')} {r.get('role','')}".lower()) else 0.0
        score = 100 * (W_SEM * s + W_SIG * sig_score + W_ROLE * role_score)
        why = f"Semantic similarity {s:.2f}; school-ecosystem signals: {', '.join(hit) or 'none'}"
        why += "; senior/advisory role fit" if role_score else ""
        rows.append({**r.to_dict(), "score": round(score, 1), "semantic": round(float(s), 3),
                     "signals": hit, "why": why})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return out, method
