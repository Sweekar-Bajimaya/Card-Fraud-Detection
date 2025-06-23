import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Load model and scaler
model = joblib.load('./models/model_rf.pkl')
scaler = joblib.load('./models/scaler.pkl')

st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detection App")

# File upload
uploaded_file = st.file_uploader("Upload transaction data (CSV)", type=["csv"])

# Helper function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

# Helper function to plot ROC curve
def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(fig)

# Helper function to plot Precision-Recall curve
def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    st.pyplot(fig)

# When file is uploaded
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Preserve original if 'Class' exists
    if 'Class' in data.columns:
        y_true = data['Class']
        data = data.drop('Class', axis=1)
    else:
        y_true = None

    # Scale 'Amount' and 'Time'
    if 'Amount' in data.columns and 'Time' in data.columns:
        data['scaled_amount'] = scaler.transform(data['Amount'].values.reshape(-1, 1))
        data['scaled_time'] = scaler.transform(data['Time'].values.reshape(-1, 1))
        data = data.drop(['Amount', 'Time'], axis=1)

    # Predict
    y_pred = model.predict(data)
    y_prob = model.predict_proba(data)[:, 1]

    # Show results
    data['Prediction'] = y_pred
    data['Fraud Probability'] = y_prob
    st.write("### 🔍 Prediction Results")
    st.dataframe(data)

    # Count summary
    st.write("### 📊 Summary")
    st.write(data['Prediction'].value_counts().rename({0: 'Non-Fraud', 1: 'Fraud'}))

    # Visuals (if ground truth exists)
    if y_true is not None:
        st.write("### 🧠 Model Performance Visualizations")
        plot_confusion_matrix(y_true, y_pred)
        plot_roc(y_true, y_prob)
        plot_pr_curve(y_true, y_prob)
else:
    st.info("Please upload a CSV file with transaction data.")
