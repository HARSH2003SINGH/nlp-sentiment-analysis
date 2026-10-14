# NLP-Based Sentiment Analysis of Customer Reviews

## Objective
Classify customer reviews into Negative, Neutral, and Positive sentiment.

## Notebook workflow
1. Data Loading & Understanding
2. Exploratory Data Analysis
3. Sentiment Labeling
4. Text Preparation / Preprocessing
5. VADER & TextBlob Baselines
6. Train-Test Split
7. TF-IDF Feature Extraction
8. Five Text Classification Models
9. Model Comparison & Evaluation
10. Error Analysis
11. Probability Calibration for Deployment
12. Save Model, Vectorizer & Model Metadata

## Five ML models
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Random Forest
- SGD Classifier

## Deployment model
The best model is calibrated using `CalibratedClassifierCV` before deployment. This allows the Streamlit application to show:
- Predicted sentiment: Negative / Neutral / Positive
- Prediction confidence for the selected sentiment
- Class-wise confidence for all three sentiments
- Overall model test accuracy

**Important:** A single review's percentage is a confidence estimate, not the model's overall accuracy.

## Important
Run the notebook from top to bottom using **Run All**. The notebook intentionally creates `text` before VADER/TextBlob and `clean_text` before TF-IDF, so the feature columns are available when they are needed.

If VADER is not installed:
```bash
python -m pip install vaderSentiment textblob
```

Or inside Jupyter:
```python
import sys
!{sys.executable} -m pip install vaderSentiment textblob
```

## Streamlit app
After running the notebook, start the app from this folder:
```bash
streamlit run app.py
```

The app uses:
- `model/sentiment_model.pkl`
- `model/tfidf_vectorizer.pkl`
- `model/model_metadata.json`
