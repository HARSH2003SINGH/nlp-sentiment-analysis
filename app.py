import json
import re
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="💬",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

    metadata = {}
    metadata_path = MODEL_DIR / "model_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, vectorizer, metadata


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.title("💬 Customer Review Sentiment Analyzer")
st.write("Enter a customer review to predict its sentiment.")

try:
    model, vectorizer, metadata = load_artifacts()

    if metadata.get("model_name"):
        st.caption(f"Model: {metadata['model_name']}")

    if metadata.get("accuracy") is not None:
        st.metric("Overall Model Accuracy", f"{metadata['accuracy']:.2%}")

except FileNotFoundError:
    st.error("Model files not found. Run the notebook from top to bottom first.")
    st.stop()

review = st.text_area(
    "Customer Review",
    placeholder="Example: The phone has excellent battery life and the camera quality is amazing.",
    height=140,
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        try:
            cleaned = clean_text(review)

            if not cleaned:
                st.warning("The review does not contain usable English text.")
                st.stop()

            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            class_probabilities = dict(zip(model.classes_, probabilities))
            confidence = class_probabilities[prediction]

            st.divider()
            st.subheader("Prediction")

            if prediction == "Positive":
                st.success(f"🟢 Predicted Sentiment: **Positive**")
            elif prediction == "Negative":
                st.error(f"🔴 Predicted Sentiment: **Negative**")
            else:
                st.info(f"🟡 Predicted Sentiment: **Neutral**")

            st.metric("Prediction Confidence", f"{confidence:.2%}")

            st.subheader("Sentiment Confidence")
            for label in ["Positive", "Neutral", "Negative"]:
                probability = class_probabilities.get(label, 0.0)
                st.write(f"**{label}** — {probability:.2%}")
                st.progress(float(probability))

            st.caption(
                "Prediction confidence shows how strongly the calibrated model supports each sentiment. "
                "It is different from the overall model accuracy shown above."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
