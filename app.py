from pathlib import Path
import re
import html
import unicodedata

import joblib
import streamlit as st


st.set_page_config(
    page_title="Researchly Research Abstract Classifier",
    page_icon="📄",
    layout="centered"
)


def clean_text(value):
    """Apply the same text cleaning used during model development."""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    text = re.sub(
        r"\b10\.\d{4,9}/\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = text.lower()

    text = "".join(
        character
        if character.isalnum() or character.isspace()
        else " "
        for character in text
    )

    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_pipeline():
    model_path = (
        Path(__file__).resolve().parent
        / "research_classifier.joblib"
    )

    bundle = joblib.load(model_path)
    return bundle["pipeline"]


st.title("Researchly Research Abstract Classifier")

st.write(
    "This app classifies research abstracts as either "
    "AI-related or classical research."
)

try:
    pipeline = load_pipeline()
except Exception as error:
    st.error(
        "The classifier could not be loaded. Check that "
        "research_classifier.joblib is in the same folder as app.py."
    )
    st.exception(error)
    st.stop()


with st.form("classification_form"):
    abstract = st.text_area(
        "Enter research abstract:",
        height=250,
        placeholder="Paste the research abstract here..."
    )

    submitted = st.form_submit_button("Classify Abstract")


if submitted:
    if not abstract.strip():
        st.warning("Please enter a research abstract.")
    else:
        cleaned_abstract = clean_text(abstract)

        if not cleaned_abstract:
            st.warning(
                "No usable text remains after cleaning. "
                "Please enter a research abstract containing words."
            )
        else:

            features = pipeline.named_steps["tfidf"].transform(
                [cleaned_abstract]
            )

            if features.nnz == 0:
                st.warning(
                    "The text contains no recognised model vocabulary. "
                    "Please provide a more complete research abstract."
                )
            else:
                prediction = pipeline.predict(
                    [cleaned_abstract]
                )[0]

                category_names = {
                    "ai": "artificial intelligence",
                    "classical": "classical research"
                }

                st.subheader("Prediction")
                st.success(
                    "The abstract is classified as: "
                    + category_names[prediction]
                )

st.caption(
    "Predictions support research screening and may be incorrect. "
    "Review the paper before deciding whether to include it."
)
