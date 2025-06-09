---
title: Credit Score Classifier
emoji: 💳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---
# **Credit Scorecard Project**

### **Table of Contents**
1. Project Overview  
2. Features  
3. Technologies Used  
4. Installation  
5. Usage  
6. Data  
7. Model  
8. Evaluation  
9. Future Enhancements  
10. Contributing  
11. Contact  

---

### **Project Overview**

**Business Problem**  
A credit company is experiencing high loan default rates due to inaccurate applicant assessments. The company seeks a data-driven approach to assess customers' creditworthiness and mitigate risks.

**Business Objectives**
- **Minimize Loan Default Rates**: Predict risky customers before approval.
- **Improve Assessment**: Use financial behavior and account metrics.
- **Enhance Profitability**: Approve primarily low-risk applicants.
- **Support Financial Growth**: Offer guidance or micro-loans to borderline applicants.

---

### **Features**

- **Numerical Credit Score Prediction** (300–850 scale)
- **Score Interpretation** (Poor, Fair, Good, Very Good, Exceptional)
- **Randomizer Function** to simulate different profiles
- **Visualization**: Confusion matrix, accuracy score, seaborn plot
- **Gradio Web Interface** for real-time testing

---

### **Technologies Used**

- **Python 3.11**
- **Libraries**:  
  - `pandas`, `numpy`, `scikit-learn`, `joblib`, `gradio`  
  - `seaborn`, `matplotlib`

---

### **Installation**

1. Clone the repo or upload files to your environment (e.g. Hugging Face Spaces)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Usage
Launch the app locally:

bash
Copy
Edit
python app.py
Or deploy to Hugging Face Spaces with this app.py and requirements.txt.

Data
The dataset is sourced from Kaggle and contains anonymized financial behavior for credit applicants, including:

Account Balances (Min/Max/Net)

Transfers and Income Streams

Penalties, Fees, and Payment Behavior

Ratios such as Savings/Salary/Expenses

Model
Trained as a regression model using RandomForestRegressor

Outputs a credit score between 300–850

Score is mapped to one of 5 categories:

Score Range	Credit Rating	Interpretation
300–579	Poor	High risk of default
580–669	Fair	Subprime borrower, elevated risk
670–739	Good	Acceptable risk
740–799	Very Good	Low risk, favorable terms
800–850	Exceptional	Very low risk, best interest rates

Evaluation
The model was trained and tested using train_test_split. Below are results from test data:

Accuracy: Based on RMSE and regression prediction range evaluation

Confusion Matrix: Grouped interpretation categories

Visualization:



Results
Predicts numeric score on a 300–850 scale

Outputs both score and interpretation

Ready for API integration or UI feedback

Future Enhancements
Incorporate deep learning or hybrid stacking models

Train on larger real-world credit datasets

Add user-specific score history tracking

Secure endpoint deployment with Hugging Face Token Auth

Contributing
Contributions are welcome!

Fork the repo

Create a feature branch

Submit a pull request 

Contact
Maintainer: @Mphoentle