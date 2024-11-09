
# Comprehensive Analysis of Amazon Customer Reviews for Enhanced Product Insights

A comprehensive analysis of Amazon customer reviews leveraging NLP, machine learning, and deep learning techniques to gain valuable insights into product ratings, categories, and review helpfulness. This project applies a hybrid approach using Term Frequency-Inverse Document Frequency (TF-IDF) and BERT embeddings, combined with various ML and DL models, to analyze large-scale Amazon review data for enhanced product insights.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Technologies Used](#technologies-used)
- [Data Preprocessing Techniques](#data-preprocessing-techniques)
- [Embedding Generation](#embedding-generation)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Results](#results)
- [Running the Project](#running-the-project)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

This project explores Amazon US customer reviews using a combination of traditional and advanced NLP techniques. The study aims to address three main research questions (RQs):
1. **Predicting Star Ratings**: Analyzing review text to predict star ratings (1-5).
2. **Classifying Product Categories**: Categorizing reviews based on product type.
3. **Predicting Review Helpfulness**: Determining the helpfulness of reviews.

For each RQ, data preprocessing, vectorization, model selection, and visualization techniques reveal underlying trends and insights. The analysis combines feature extraction with ML and DL approaches, such as logistic regression, Naive Bayes, RNN, and LSTM, to evaluate performance on TF-IDF and BERT embeddings.

---

## Folder Structure

The repository is organized as follows:

```
Comprehensive-Analysis-of-Amazon-Customer-Reviews-for-Enhanced-Product-Insights/
│
├── data/
│   ├── amazon_reviews.csv                   # Sampled dataset (optional, include if manageable)
│   ├── bert_embeddings_star_rating.pkl       # BERT embeddings for RQ1
│   ├── bert_embeddings_product_category.pkl  # BERT embeddings for RQ2
│   ├── bert_embeddings_helpfulness.pkl       # BERT embeddings for RQ3
│
├── notebooks/
│   └── Amazon_Reviews_Insights_Analysis.ipynb  # Main Jupyter Notebook with all code
│
├── requirements.txt                           # Dependencies for the project
└── README.md                                  # Project documentation
```

---

## Technologies Used
- **Python**: Core programming language.
- **NLP Libraries**: NLTK for text preprocessing and tokenization.
- **Machine Learning**: Scikit-learn for TF-IDF vectorization and ML models.
- **Deep Learning**: PyTorch for RNN, LSTM, GRU models; Transformers for BERT embeddings.
- **Data Visualization**: Matplotlib for generating visual insights.

---

## Data Preprocessing Techniques
- **Lowercasing**: Ensures uniformity by converting text to lowercase.
- **Tokenization**: Splits text into individual words for further processing.
- **Stopword Removal**: Filters out common but uninformative words.
- **Lemmatization**: Reduces words to their base forms for improved model generalization.
