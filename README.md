# AI Document Risk & Compliance Classifier

## Overview

AI Document Risk & Compliance Classifier is a machine learning project that automatically classifies business documents into different risk categories and flags uncertain cases for manual review.

## Features

* Document risk classification
* Categories:

  * Low Risk
  * Medium Risk
  * High Risk
  * Compliance Required
* Confidence score generation
* Manual review recommendation for low-confidence predictions
* Simple Streamlit web interface

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* TF-IDF Vectorization
* Naive Bayes Classification

## Project Workflow

1. Load document dataset
2. Convert text into TF-IDF features
3. Train machine learning model
4. Predict document risk category
5. Calculate confidence score
6. Flag uncertain documents for manual review

## Installation

Install required packages:

```bash
pip install pandas scikit-learn streamlit
```

## Run the Training Script

```bash
py train.py
```

## Run the Application

```bash
py -m streamlit run app.py
```

## Example Output

Risk Category: High Risk

Confidence: 33.62%

Manual Review Required

## Future Enhancements

* PDF document upload
* Larger training dataset
* Advanced machine learning models
* Interactive dashboard and analytics
* Explainable AI features

## Author

Dhanashree Pande
BCA Student
