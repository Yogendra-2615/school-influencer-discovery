import pandas as pd, streamlit as st
from ranker import rank, profile_text

st.set_page_config(page_title="School Influencer Discovery", layout="wide")
st.title("School Ecosystem Influencer Discovery")
st.caption("Reference profile → AI analysis → similar profile discovery → ranking → explained results")

up = st.sidebar.file_uploader("Profiles CSV (lawfully sourced)", type="csv")
path = up or "data/profiles.csv"
try:
    df = pd.read_csv(path).fillna("")
except FileNotFoundError:
    st.error("Add data/profiles.csv (see README for schema and compliant data sources)."); st.stop()
top_n = st.sidebar.slider("Results", 5, 50, 25)

mode = st.radio("Reference profile", ["Pick from dataset", "Paste profile text"], horizontal=True)
exclude = None
if mode == "Pick from dataset":
    name = st.selectbox("Reference person", df["name"])
    ref = df[df["name"] == name].iloc[0]
    ref_text, exclude = profile_text(ref), ref["url"]
else:
    ref_text = st.text_area("Headline, roles, summary, experience", height=180)

if st.button("Discover", type="primary") and ref_text.strip():
    res, method = rank(df, ref_text, exclude, top_n)
    st.success(f"{len(res)} profiles ranked using {method} + signal/role scoring")
    st.dataframe(res[["name", "url", "role", "organization", "location", "score", "why"]],
                 column_config={"url": st.column_config.LinkColumn("Source URL")}, use_container_width=True)
    for _, r in res.iterrows():
        with st.expander(f"{r['score']} · {r['name']} — {r['role']} @ {r['organization']}"):
            st.write(r["why"]); st.write(r["headline"]); st.caption(r["experience"])
    st.download_button("Download CSV", res.to_csv(index=False), "results.csv")
