OVERVIEW    
Customer churn is a critical issue for businesses, particularly in subscription-based industries. This project uses Machine Learning techniques to predict whether a customer is likely to churn based on their historical data.

📊 DATASET   
Source: Telco Customer Churn Dataset (or mention your dataset source)
Features: Customer demographics, account details, service usage, tenure, contract type, payment method, and monthly charges.
Target Variable: "Churn" (Binary: Yes/No)
🛠️ TECHNOLOGIES USED   
Programming Language: Python 🐍
Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn
Machine Learning Models: Random Forest, Logistic Regression, XGBoost (Ensemble Learning)
Resampling Techniques: SMOTE (Synthetic Minority Over-sampling Technique)
Data Processing: Feature Engineering, Standardization, One-Hot Encoding
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
🚀 PROJECT WORKFLOW   
Data Preprocessing

Handle missing values
Convert categorical variables into numerical form
Standardize numerical features
Exploratory Data Analysis (EDA)

Visualizing churn vs. non-churn customers
Identifying correlations and patterns in features
Handling Imbalanced Data

Applied SMOTE to balance churned and non-churned classes
Model Training & Evaluation

Split dataset into train (80%) and test (20%)
Trained multiple ML models and compared performance
Used Ensemble Learning (Random Forest + XGBoost) for higher accuracy
Final Results

Best Model: Random Forest + SMOTE
Churn Class F1-score: 0.79 (Significant improvement after resampling)
Overall Accuracy: 81%
📈 RESULTS AND INSIGHTS    
The churn prediction model improved significantly after balancing the dataset using SMOTE.
Monthly charges, tenure, and contract type were the most influential features affecting churn.
Ensemble models outperformed single models like logistic regression, showing a better recall score for churned customers.
