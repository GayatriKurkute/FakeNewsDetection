# 📰 Fake News Detection using Natural Language Processing (NLP)

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline designed to classify news articles as **Real** or **Fake** based on their textual content. 

This project preprocesses unstructured textual data, extracts linguistic features using **TF-IDF Vectorization**, and trains classification models (such as **PassiveAggressive Classifier** and **Logistic Regression**) to identify misinformation.

---

## 📌 Project Overview

With the rapid spread of digital information across social media and news platforms, automated detection of misleading or fabricated news has become crucial. 

This project aims to detect fake news by:
1. Cleaning and normalizing raw textual data (removing noise, stop words, and punctuation).
2. Converting unstructured text into numerical vector representations.
3. Training and evaluating machine learning classifiers to accurately distinguish between authentic reporting and fake news.
4. Providing a lightweight interface (via **Streamlit** / **Flask**) for real-time predictions.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.8+
* **Natural Language Processing:** `NLTK` / `spaCy`, `Scikit-learn` (TF-IDF Vectorizer)
* **Machine Learning Models:** `Scikit-learn` (PassiveAggressiveClassifier, Logistic Regression, Naïve Bayes, Random Forest)
* **Data Processing & Analysis:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Deployment / Web Interface:** `Streamlit` / `Flask`

---

## ⚙️ Methodology & Pipeline