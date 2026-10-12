# Emotion Classification Using Machine Learning

## Project Overview

This project focuses on classifying emotions expressed in text comments using Natural Language Processing (NLP) and machine learning techniques.

The objective is to preprocess textual data, convert text into numerical features using TF-IDF, train two classification models, and compare their performance using appropriate evaluation metrics.

## Dataset

- **File name:** `nlp_dataset.csv`
- **Number of records:** 5,937
- **Input feature:** `Comment`
- **Target variable:** `Emotion`
- **Emotion categories:** Anger, Fear, and Joy

The dataset contains text comments labelled with their corresponding emotions.

**Dataset source:** [Google Drive Dataset](https://drive.google.com/file/d/1HWczIICsMpaL8EJyu48ZvRFcXx3_pcnb/view?usp=drive_link)

## Technologies and Libraries

- Python
- Pandas and NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Jupyter Notebook / Google Colab

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the dataset using Pandas.
- Examined the dataset dimensions, column names, and data types.
- Checked for missing values and duplicate records.
- Explored the distribution of emotion labels.

### 2. Text Preprocessing

The following preprocessing techniques were applied:

- **Lowercasing:** Converted text to lowercase to ensure consistent word representation.
- **Text cleaning:** Removed URLs, punctuation, numbers, and unnecessary special characters.
- **Tokenization:** Split text into individual words using NLTK.
- **Stopword removal:** Removed common English stopwords while preserving important negation words such as "not", "no", "nor", and "never".

These steps help reduce noise and create a more consistent representation of textual data.

### 3. Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was used to transform text into numerical features.

The vectorizer was configured with a maximum of 5,000 features and an n-gram range of (1, 2), allowing it to capture both individual words and two-word phrases.

The dataset was split into 80% training data and 20% testing data, using stratified sampling. The TF-IDF vectorizer was fitted only on the training data to prevent data leakage.

### 4. Model Development

Two machine learning models were trained:

- **Multinomial Naive Bayes:** A probabilistic classifier commonly used for text classification.
- **Linear Support Vector Machine (SVM):** A classifier that learns decision boundaries between emotion categories and works well with high-dimensional, sparse text features.

### 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Classification reports
- Confusion matrices

## Results

| Evaluation Metric | Naive Bayes | Linear SVM |
|---|---:|---:|
| Accuracy | 90.91% | 94.61% |
| Macro Precision | 90.91% | 94.61% |
| Macro Recall | 90.92% | 94.61% |
| Macro F1-score | 90.91% | 94.61% |

*Note: The values above are rounded from the evaluation output.*

## Conclusion

Both models performed well on the emotion classification task. Multinomial Naive Bayes achieved an accuracy of 90.91%, while Linear SVM achieved 94.61%.

Linear SVM outperformed Naive Bayes by 3.70 percentage points in accuracy and macro F1-score. Its performance demonstrates its suitability for classification tasks involving high-dimensional TF-IDF features.

Therefore, Linear SVM was selected as the best-performing model for this experiment.




**Project:** Emotion Classification Using NLP and Machine Learning
