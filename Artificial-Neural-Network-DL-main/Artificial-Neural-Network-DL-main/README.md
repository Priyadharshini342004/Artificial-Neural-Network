# Artificial-Neural-Network-DL
## 🧠 Customer Churn Prediction using Artificial Neural Network (ANN)

This project predicts whether a bank customer will exit (churn) or stay using a trained Artificial Neural Network (ANN) model built with TensorFlow/Keras.
The application is deployed with Streamlit for easy user interaction.

## 🚀 Project Overview

Customer churn refers to customers leaving a service or company.
Predicting churn helps banks retain customers and improve business strategies.

This project includes:

A trained ANN model (churn_model.h5)

Data preprocessing pipeline (preprocessor.pkl)

Interactive Streamlit web app (app.py) for live predictions

## APP:
https://annchurnprediction1.streamlit.app/

## 🧩 Tech Stack
Category	Tools/Frameworks
Programming Language	Python
Deep Learning	TensorFlow / Keras
Data Processing	Pandas, NumPy, Scikit-learn
Visualization	Matplotlib, Seaborn
Web App	Streamlit
Model Serialization	Pickle (.pkl), HDF5 (.h5)

## Visualization:

### 📊 Correlation Heatmap Visualization
<img width="1008" height="782" alt="image" src="https://github.com/user-attachments/assets/ea8e38e1-75f2-4806-a791-9607f3a7ceae" />

### Skewness calculated before and after Yeo-Johnson normalization
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/d840f6f6-f0dd-4159-ac7d-7aeee2e9f4dd" />

### Outliers:
<img width="990" height="390" alt="image" src="https://github.com/user-attachments/assets/f50c601f-3978-4787-9ac5-63ff30d9fafe" />

### Plot Accuracy and Loss Curves
<img width="1010" height="470" alt="image" src="https://github.com/user-attachments/assets/b11ba5a7-2104-4df1-82b9-0df8bc1c0fdd" />

### ROC curve:
<img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/455f5cd1-789c-4d74-9554-047864ff56f5" />


## 📂 Project Structure
📁 Customer_Churn_ANN/
│
├── app.py                     # Streamlit web app
├── churn_model.h5             # Trained ANN model
├── preprocessor.pkl           # Scaler + encoder pipeline
├── data.csv                   # (Optional) Dataset used for training
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

## ⚙️ How to Run the Project

1️⃣ Clone this repository
git clone https://github.com/yourusername/customer-churn-ann.git
cd customer-churn-ann

2️⃣ Create and activate a virtual environment
python -m venv env
env\Scripts\activate      # On Windows
# OR
source env/bin/activate   # On Mac/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py


Then open the link (usually http://localhost:8501) in your browser 🌐.

## 🧮 Model Details

Input Features:
CreditScore, Geography, Gender, Age, Tenure, Balance,
NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

Target Variable:
Exited

0 → Customer stays

1 → Customer churns (leaves the bank)

ANN Architecture:

Input Layer: 10 neurons

Hidden Layers: 2 Dense layers with ReLU activation

Output Layer: 1 neuron with Sigmoid activation

Optimizer: Adam

Loss Function: Binary Crossentropy

Metric: Accuracy

## 📊 Example Output (from Streamlit App)
Input	Output
Credit Score: 619, Age: 42, France, Female	Churn Probability: 0.27 → Low Chance of Churn (Exited = 0)

🟢 Low Chance of Churn → Customer likely to stay
🔺 High Chance of Churn → Customer likely to leave

## 📈 Evaluation Metrics
Metric	Value (example)
Accuracy	86%
Precision	83%
Recall	78%
F1-Score	80%

## 🧠 Model Interpretation

The ANN learns complex patterns from customer attributes to estimate churn probability.

A Sigmoid output gives a probability between 0 and 1.

> 0.5 → Predicts churn (Exited = 1)

< 0.5 → Predicts retention (Exited = 0)
