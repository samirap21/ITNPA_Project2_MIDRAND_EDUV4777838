# Researchly: Research Paper Classification and NLP Analysis

**ITNPA4-34 | Natural Language Processing with Python | Eduvos**  
**Author:** Samira Patel  
**Qualification:** BSc Honours in Data Science

## Overview

Researchly is a natural language processing project for organising and exploring academic research. It combines a supervised research-abstract classifier with text summarisation and topic modelling. A Streamlit interface allows users to paste an abstract and receive a prediction of whether it belongs to **AI-related** or **classical** research.

The project covers four parts: preparing research abstracts and establishing a baseline, comparing and tuning classification algorithms, summarising a collection of research articles, and extracting research themes using Latent Dirichlet Allocation (LDA).

## Key results

| Measure | Result |
|---|---:|
| Original abstract records | 1,138 |
| Records after cleaning and deduplication | 1,121 |
| Training / test records | 896 / 225 |
| TF-IDF features | 10,000 |
| Logistic regression baseline test accuracy | 90.22% |
| Selected Linear SVM test accuracy | **91.56%** |
| Selected Linear SVM test macro-F1 | **0.8669** |
| Research articles analysed | 5 |
| LDA topics configured | 3 |

**Model selection:** The project compares logistic regression, Linear SVM, multinomial Naive Bayes, complement Naive Bayes, and random forest. Linear SVM provided the highest mean cross-validation macro-F1 among the tested models. Grid search selected an SVM `C` of `0.5`, TF-IDF unigrams and bigrams, and `min_df=3`. Tuning increased cross-validation macro-F1 by approximately `0.0005`, but **did not improve the held-out test metrics** relative to the untuned SVM. These results are taken from the saved notebook outputs and have not been independently rerun.

## Methods

### 1. Abstract classification
- Load `research_abstracts.json`, containing DOI, Abstract and Label columns.
- Clean text and remove duplicate abstracts, checking for label conflicts.
- Create stratified training and test partitions and TF-IDF features.
- Evaluate a logistic regression baseline and conduct cross-validation.

### 2. Model comparison and deployment
- Compare four alternative classifiers with the baseline.
- Tune the Linear SVM and TF-IDF settings using `GridSearchCV`.
- Evaluate held-out predictions with accuracy, balanced accuracy, macro-F1, classification reports and confusion matrices.
- Save the selected text-classification pipeline as `research_classifier.joblib` and expose it through Streamlit.

### 3. Research-document analysis
- Read five `.txt` articles from `research_corpus.zip`.
- Inspect and preprocess article text.
- Produce an extractive summary of the first article using TF-IDF-based sentence scoring.
- Analyse themes in the remaining four documents using non-negative matrix factorisation (NMF) and create charts and tables.

### 4. Topic modelling
- Fit a three-topic LDA model to the five-document corpus.
- Visualise document-topic proportions and top topic words.
- Generate an interactive intertopic-distance map with `pyLDAvis`.

## Repository contents

```text
ITNPA_Project/
├── ITNPA4-34_Project1_Midrand_EDUV4777838.ipynb
├── app.py
├── research_classifier.joblib
├── research_abstracts.json
├── research_corpus.zip
├── question3_outputs/
│   ├── article_1_summary.txt
│   ├── article_1_summary_scores.png
│   ├── article_1_selection_details.csv
│   ├── aggregate_topic_contributions.csv
│   ├── aggregate_topic_heatmap.png
│   ├── aggregate_topic_terms.csv
│   └── aggregate_topic_terms.png
├── question4_outputs/
│   ├── lda_document_topic_distribution.png
│   ├── lda_document_topic_percentages.csv
│   ├── lda_interactive_map.html
│   ├── lda_top_words.csv
│   └── lda_top_words.png
├── requirements.txt              # To be added
├── .gitignore                     # To be added
└── README.md
```

## Run the classifier locally

**Requirements:** Python and pip. Python 3.13.9, scikit-learn 1.7.2 and Streamlit 1.51.0 were used during development. The saved model uses Python's joblib/pickle format: only load it from a trusted repository, and use a compatible scikit-learn version.

1. Clone or download this repository, then open a terminal inside the project folder.
2. Create and activate a virtual environment (Windows):

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Once `requirements.txt` has been added, install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the application:

   ```powershell
   python -m streamlit run app.py
   ```

5. Open the URL Streamlit prints (typically `http://localhost:8501`), paste an abstract and select **Classify Abstract**.

`app.py` expects `research_classifier.joblib` **in the same directory**. The model is a saved pipeline, so training is not required to use the existing app.

### Reproduce notebook results

Open the notebook in Jupyter and execute cells in order. **Current portability limitation:** some notebook cells still contain absolute paths referencing the original Windows project directory, and package-installation cells are embedded in the notebook. These must be changed to project-relative paths and dependencies installed before a clean run on another machine. Reproduction from a fresh environment has **not yet been verified**.

## Outputs

The `question3_outputs/` folder includes the first-article summary and charts for the aggregate document analysis. The `question4_outputs/` folder includes LDA topic summaries, plots and the interactive `lda_interactive_map.html`. Open the HTML file in a web browser to explore the topic map.

## Limitations

- Results are based on the supplied abstracts and a corpus of five research articles; generalisation to other fields or document collections is not established.
- The classes are imbalanced, so accuracy alone does not capture all aspects of model performance.
- Hyperparameter tuning produced only a minor cross-validation gain and no test-set gain over the original SVM.
- The interface is intended to assist research screening, not replace human review.

## Academic use

Developed as coursework for **ITNPA4-34**. Follow Eduvos submission, attribution and AI-disclosure requirements. Publish supplied datasets or course material only if you have permission.
