import re
from datetime import datetime

import joblib
import streamlit as st
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words("english"))
negations = {"no", "not", "nor", "n't", "never", "none", "cannot"}
custom_stopwords = stop_words - negations
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in custom_stopwords and len(t) > 1]
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in tagged]
    return " ".join(lemmas)

@st.cache_resource
def load_model():
    return joblib.load("sentiment_analysis_best_model.joblib")

model = load_model()

# --- Page setup ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="🌸", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #fff0f5 0%, #f2fbf5 60%, #e6f7ec 100%); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2b2438 0%, #241f30 100%);
}
section[data-testid="stSidebar"] * { color: #f3e8ff !important; }
h1, h2, h3 { color: #4a3b52 !important; }
.stat-card {
    border-radius: 16px;
    padding: 1.1em 1.3em;
    color: #33263a;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}
.card-pink   { background: #ffd6e7; }
.card-green  { background: #cdf2dd; }
.card-yellow { background: #fdf1c7; }
.card-mint   { background: #d7f5ef; }
.stat-label { font-size: 0.85em; opacity: 0.75; margin-bottom: 0.2em; }
.stat-value { font-size: 1.9em; font-weight: 800; }
div.stButton > button {
    background: linear-gradient(90deg, #f472b6, #34d399);
    color: white; border: none; border-radius: 12px;
    padding: 0.6em 1.5em; font-weight: 600;
}
div.stButton > button:hover { opacity: 0.9; color: white; }
.result-box {
    padding: 1.2em; border-radius: 14px; margin-top: 0.8em;
    font-size: 1.2em; font-weight: 600; text-align: center;
}
.positive { background: #cdf2dd; color: #166534; }
.negative { background: #ffd6e7; color: #9d174d; }
.history-item {
    background: #ffffffaa; border-radius: 12px; padding: 0.7em 1em;
    margin-bottom: 0.5em; font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# --- Session state: history of analyzed reviews ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🌸 Sentiment AI")
    st.caption("TF-IDF + SVM")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("📊 Dashboard")
    st.markdown("📝 History")
    st.markdown("ℹ️ About")
    st.markdown("---")
    st.caption("Model: Linear SVM")
    st.caption("Vectorizer: TF-IDF (1–2 grams)")
    st.caption("Trained on 2,000 reviews")

# --- Header ---
st.title("🌸 Sentiment Analysis Dashboard")
st.caption("Type a review below and watch it get classified in real time.")

# --- Stat cards ---
total = len(st.session_state.history)
pos_count = sum(1 for h in st.session_state.history if h["label"] == "pos")
neg_count = total - pos_count
avg_conf = (sum(h["confidence"] for h in st.session_state.history) / total) if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-card card-yellow"><div class="stat-label">Total Analyzed</div><div class="stat-value">{total}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card card-green"><div class="stat-label">Positive</div><div class="stat-value">{pos_count}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card card-pink"><div class="stat-label">Negative</div><div class="stat-value">{neg_count}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card card-mint"><div class="stat-label">Avg Confidence</div><div class="stat-value">{avg_conf:.0%}</div></div>', unsafe_allow_html=True)

st.markdown("")

left, right = st.columns([1.3, 1])

# --- Left: input + analyze ---
with left:
    st.subheader("Analyze a Review")
    review_text = st.text_area("Your Review", placeholder="Type a product or movie review here...", height=140, label_visibility="collapsed")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Positive example"):
            review_text = "This product exceeded my expectations, I absolutely love it!"
    with b2:
        if st.button("Negative example"):
            review_text = "Terrible quality, it broke after one day and I am not happy at all."

    if st.button("Analyze Sentiment", type="primary"):
        if not review_text or not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = preprocess(review_text)
            prediction = model.predict([cleaned])[0]
            try:
                proba = model.predict_proba([cleaned])[0]
                confidence = float(max(proba))
            except AttributeError:
                confidence = 1.0

            st.session_state.history.insert(0, {
                "text": review_text,
                "label": prediction,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S"),
            })

            if prediction == "pos":
                st.markdown(f'<div class="result-box positive">😊 Positive — confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box negative">😠 Negative — confidence: {confidence:.1%}</div>', unsafe_allow_html=True)

            word_count = len(review_text.split())
            if word_count < 5:
                st.caption("⚠️ Very short reviews can be less reliable — the model has fewer words to learn from.")

            word_count = len(review_text.split())
            if word_count < 5:
                st.caption("⚠️ Very short reviews and slang can be less reliable — the model has little text to go on.")

    if total > 0:
        st.markdown("")
        st.subheader("Sentiment Breakdown")
        chart_df = pd.DataFrame({"Sentiment": ["Positive", "Negative"], "Count": [pos_count, neg_count]}).set_index("Sentiment")
        st.bar_chart(chart_df, color="#f472b6")

# --- Right: history list ---
with right:
    st.subheader("Recent Reviews")
    if not st.session_state.history:
        st.caption("No reviews analyzed yet — try one on the left.")
    else:
        for item in st.session_state.history[:8]:
            emoji = "😊" if item["label"] == "pos" else "😠"
            preview = item["text"][:70] + ("…" if len(item["text"]) > 70 else "")
            st.markdown(
                f'<div class="history-item">{emoji} <b>{item["label"].upper()}</b> '
                f'({item["confidence"]:.0%}) · {item["time"]}<br>{preview}</div>',
                unsafe_allow_html=True,
            )
