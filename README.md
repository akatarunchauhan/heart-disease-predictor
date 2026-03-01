# 🫀 Heart Disease Predictor

End-to-end ML project predicting heart disease from clinical parameters.

## Stack
- **Model**: Voting Ensemble (LR + SVM + RF + Gradient Boosting)
- **Frontend**: Streamlit
- **Dataset**: Cleveland UCI Heart Disease Dataset (303 samples, 13 features)

## Setup

1. Clone the repo
```bash
   git clone https://github.com/akatarunchauhan/heart-disease-predictor
   cd heart-disease-predictor
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

3. Generate the model bundle (run the notebook top to bottom)
```bash
   jupyter notebook notebook/end-to-end-heart-disease-classification.ipynb
```
   This saves `model/heart_disease_bundle.pkl`

4. Run the app
```bash
   streamlit run app.py
```

## Features
- 6 trained models with 5-fold cross-validation scores
- Feature engineering (age×thalach, chol×age, age groups)
- Confidence scores, risk flags, confusion matrix, feature importances
- Model bundle exported from notebook — zero retraining on app startup
