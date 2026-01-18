 Comprehensive Analysis of Amazon Customer Reviews for Enhanced Product Insights

This repository contains the **experimental implementation and analysis** corresponding to the research paper:

**A Comprehensive Framework for Multi-Aspect Analysis of Amazon Customer Reviews Using Machine and Deep Learning**  
DOI: https://doi.org/10.21203/rs.3.rs-7979960/v1

The work investigates multiple aspects of Amazon U.S. customer reviews using machine learning (ML) and deep learning (DL) models. The framework focuses on **star rating prediction**, **product category classification**, and **review helpfulness analysis**, integrating traditional feature representations with contextual language embeddings.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Objectives](#objectives)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Embedding Generation](#embedding-generation)
  - [Machine Learning Models](#machine-learning-models)
  - [Deep Learning Models](#deep-learning-models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
With the growing influence of online reviews on purchasing decisions, this project aims to analyze large-scale Amazon customer reviews to provide valuable insights for e-commerce platforms. The study focuses on:
1. Predicting the star rating from review text.
2. Classifying the product category.
3. Assessing the helpfulness of reviews.

## Dataset
The dataset used in this project is the **Amazon US Customer Reviews Dataset**, available on Kaggle:
- [Amazon US Customer Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset/data?select=amazon_reviews_multilingual_US_v1_00.tsv)
  
Download the dataset from Kaggle and sample it based on the project requirements:
- **Full Dataset**: The dataset contains millions of reviews across various product categories. For computational efficiency, a random sample of **200,000 reviews** is recommended.

## Objectives
1. **Star Rating Prediction**: Develop a model to predict the review's star rating based on the content.
2. **Product Category Classification**: Classify reviews into relevant product categories.
3. **Helpfulness Prediction**: Predict whether a review is marked helpful by other users.

## Methodology
### Data Preprocessing
- **Text Processing**: Tokenization, stopword removal, lemmatization, and lowercasing for consistency.
- **Embedding Generation**: TF-IDF for term significance and BERT embeddings for contextual semantics.

### Embedding Generation
The project leverages both TF-IDF and BERT embeddings for feature extraction:
- **BERT Embeddings**: Capturing contextual semantics using the `bert-base-uncased` model.
- **TF-IDF Vectors**: Capturing syntactic frequency patterns with a maximum of 5,000 features.

### Machine Learning Models
We implemented and compared multiple ML models:
- **Logistic Regression**
- **Naive Bayes**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**

### Deep Learning Models
To handle sequential data, the following DL models were employed:
- **Recurrent Neural Network (RNN)**
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**

### Evaluation Metrics
- **F1-Score**: Used for star rating and product category classification tasks due to class imbalance.
- **Accuracy**: Used for helpfulness prediction due to a balanced dataset.

## Results
Results for each research question are stored and discussed in `notebooks/Amazon_Reviews_Insights_Analysis.ipynb`. We observe that BERT embeddings often outperform TF-IDF, especially with RNN-based models, due to their context-aware nature.

## Visualizations
The project includes several visualizations for data exploration:
- **Word Clouds** for frequently occurring terms in various star ratings and helpfulness categories.
- **Bar Charts** showing distribution patterns for star ratings and product categories.

## Requirements
The required Python packages are listed in `requirements.txt`.

## Usage
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Citation

If you use this work, please cite:

```bibtex
@article{chakraborty2025comprehensive,
  title   = {A Comprehensive Framework for Multi-Aspect Analysis of Amazon Customer Reviews Using Machine and Deep Learning},
  author  = {Chakraborty, Pratik and Ameen, Shaik Nurul and Nayak, Ashalatha},
  year    = {2025},
  journal = {Research Square},
  doi     = {10.21203/rs.3.rs-7979960/v1}
}
```
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



