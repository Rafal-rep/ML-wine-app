# Wine Quality Prediction App

Live demo: [https://ww-app-d268e1d36d7c.herokuapp.com/]

This project is a complete machine learning pipeline with a web interface for predicting the quality of red and white wines based on their physicochemical properties.

It consists of:
- Model training and evaluation in Jupyter notebooks (white and red wine separately)
- Exported `.pkl` models and scalers using `joblib`
- A FastAPI-based web app for local or cloud deployment

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data](#data)
- [Modeling](#modeling)
- [Web Application](#web-application)
- [How to Run Locally](#how-to-run-locally)

## Overview

The goal of this project is to predict wine quality based on physicochemical inputs using machine learning.  
The app allows users to input values manually and receive a prediction categorized as:
- Zła jakość (Bad)
- Średnia jakość (Medium)
- Dobra jakość (Good)

## Features

- Separate models for red and white wines
- Data preprocessing using `StandardScaler`
- Classification using `RandomForestClassifier`
- FastAPI web interface with HTML templates
- Visual form interface styled with CSS

## Data

The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).  
It includes physicochemical attributes such as:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

Target: wine quality (score from 0 to 10)

## Modeling

The training was performed in separate notebooks:
- `RED.ipynb`: preprocessing and training model for red wine
- `WHITE.ipynb`: same for white wine

Each model was saved using `joblib`:
- `redwine_rf.pkl`
- `whitewine_rf.pkl`
- with their respective scalers:
  - `scaler_red.pkl`
  - `scaler_white.pkl`

## Web Application

The FastAPI app (`main.py`) does the following:
1. Loads models and scalers during startup
2. Serves a form (`index.html`) where users:
   - Select wine type (red or white)
   - Enter 11 numerical features
3. Transforms and scales input data
4. Returns a prediction mapped to a quality label

### Mapping:
- **White wine**:
  - 0 → Zła jakość (Low quality)
  - 1 → Średnia jakość (Medium quality)
  - 2 → Dobra jakość (High quality)

- **Red wine**:
  - 0 → Zła jakość (Low quality)
  - 1 → Dobra jakość (High quality)

## How to Run Locally

### Prerequisites
- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

- If needed, install missing modules manually:

```bash
pip install scikit-learn fastapi uvicorn joblib numpy jinja2 python-multipart
```

### Run the App

```bash
cd deployment
python -m uvicorn main:app --reload
```

Then go to `http://127.0.0.1:8000` in your browser to use the app.

