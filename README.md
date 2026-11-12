# School Ecosystem Influencer Discovery (EDXSO Assignment 3)

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Deploy: push to GitHub → share.streamlit.io → select `app.py` (free live URL).

## Data (compliance)
No LinkedIn scraping, login bypass or CAPTCHA circumvention. Put real profiles in `data/profiles.csv`
(schema: `data/profiles.template.csv`) from lawful sources only, e.g.:
- profiles you manually curate from public pages (conference speaker lists, school/trust leadership pages, published directories) with the URL kept for traceability;
- a licensed/authorised profile-data API or LinkedIn's official partner programs;
- members' own consented data exports.
Collect 60+ rows with a mix of relevant and irrelevant people so ranking is meaningful. Do not fabricate profiles.
Add a `data/SOURCES.md` listing where each batch came from.

## Technical note
**Architecture:** CSV store → `ranker.py` (analysis + scoring) → Streamlit UI (`app.py`).
**AI analysis:** the reference profile's text is checked for school-ecosystem signals (leadership, advisory, multi-school network, K-12 context, entrepreneurship); these define what "relevant" means for that reference.
**Similarity:** sentence embeddings (all-MiniLM-L6-v2) with cosine similarity capture meaning beyond exact keywords (e.g. "headmistress" ≈ "principal"). TF-IDF is an automatic fallback.
**Ranking:** `score = 100 × (0.55·semantic + 0.30·signal overlap with reference + 0.15·senior/advisory role fit)`. Each result's explanation lists the semantic score and matched signals.
**Limitations:** only as good as the dataset; regex signals are English/India-K12 centric; no connection-graph data; weights are hand-set, not learned.
**Production path:** licensed profile data API or LinkedIn partner access, vector DB (pgvector), labelled feedback from outreach to learn weights, cross-encoder re-ranking.

## Demo script (5–10 min)
1) show dataset and sources; 2) pick a reference; 3) Discover; 4) walk through scores/explanations; 5) paste-in a new reference; 6) discuss limits.
