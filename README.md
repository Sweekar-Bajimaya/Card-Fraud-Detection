# 💳 Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions using a real-world dataset from Kaggle.

## 🚀 Project Highlights

- **Highly imbalanced dataset**
- **Data preprocessing** with scaling
- **Model training** using Logistic Regression and Random Forest
- **Evaluation** using Confusion Matrix, ROC & Precision-Recall Curves
- **SMOTE oversampling** to handle class imbalance
- 📊 **Interactive Streamlit App** for predictions and visualizations

---

## 📎 Dataset
The Dataset was taken from Kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

---

## 📁 Project Structure

| File/Folder         | Purpose                                     |
|---------------------|---------------------------------------------|
| `notebook/`         | Colab notebook with full pipeline         |
| `streamlit_app/`    | Streamlit app with trained model            |
| `data/`             | Sample transaction input file (CSV)         |
| `requirements.txt`  | All required packages for the project       |

---

## ⚙️ Technologies Used

- Python, Pandas, Scikit-learn
- Streamlit for deployment
- SMOTE for handling imbalance
- Matplotlib & Seaborn for visualization

---

## 🧪 Model Performance

| Metric        | Random Forest | Logistic Regression |
|---------------|---------------|---------------------|
| ROC AUC       | 0.97          | 0.97                |
| Precision     | High          | Lower               |
| Recall        | Good          | Higher              |
| AUC-PR        | 0.8675        | 0.7249              |

✅ Random Forest was chosen for final deployment due to higher precision and fewer false alarms.

---

## 📈 Visualization 
![image](https://github.com/user-attachments/assets/3e798623-38dc-4a50-bf45-7b4512c9d338)
![image](https://github.com/user-attachments/assets/e6b06573-e758-42b1-9949-71c561a74a5d)
![image](https://github.com/user-attachments/assets/33643b37-f40d-46d9-8319-0a36691e769b)
![image](https://github.com/user-attachments/assets/74935d71-d18b-4380-94c6-018e0b534b98)
![image](https://github.com/user-attachments/assets/27dc4827-1e77-4e98-81fb-217469228be9)



