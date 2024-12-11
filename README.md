# Data-Driven Customer Churn Forecasting: Enhancing Retention Strategies for Business Growth

This repository contains the code and resources for a Customer Churn Prediction project. The project aims to predict customer churn using various machine learning models, comparing their performance, and identifying the most effective approach. The work involves extensive data preprocessing, exploratory data analysis, model building, and evaluation.
# Project overview
Customer churn prediction is a critical task for businesses to identify customers who are likely to leave their service and take proactive measures to retain them. This project utilizes data analysis and machine learning techniques to classify customers based on their likelihood to churn.
## Table of Contents
- [Installation](#installation)
- [Key features](#Keyfeatures)
- [data](#data)
- [Model Performance Metrics](modelperformancemetrics)
- [Results](#results)
- [Challenges and Limitations](#Challenges and Limitations)
- [Recommendations and Future Scope](#RecommendationsandFutureScope)
-  [future work](#futurework)

## Installation
**application** | **libraries**
--- | --- 
handle table-like data and matrices | pandas, numpy 
visualisation | plotly, seaborn, missingno 
classification models | sklearn

## Key features

* [x] Exploratory Data Analysis (EDA): Analyze the data to uncover trends, patterns, and potential predictors of churn.
* [x] Data Preprocessing: Convert categorical variables to numerical formats and handle missing values.
* [x] Modeling: Build and evaluate multiple machine learning models, including:
   Logistic Regression,
   Decision Trees,
   Random Forest,Gradient Boosting.
* [x] Performance Metrics: Use metrics such as accuracy, ROC-AUC, and confusion matrices for model evaluation.
## Data
[Kaggle link: customer churn data](https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset)

[Github link: Telco Customer Churn](https://github.com/Boppana-vyshnavi/Boppana-vyshnavi/blob/main/Churn_.csv)

### Model Performance Metrics
- Cross-Validation Metrics
  We evaluate our models using the following metrics:
  1. Accuracy
  2. Recall
  3. Precision
  4. F1 Score
  5. ROC_AUC
- Random Forest Model Metrics:
  ``` sh
  import pandas as pd
  ## Dataframe setup
  from sklearn.metrics import confusion_matrix, classification_report
  models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree']
  predictions = [log_reg_pred, rf_pred, gb_pred, dt_pred]

    for model, pred in zip(models, predictions):
      print(f"Evaluation for {model}:\n")
      print(confusion_matrix(y_test, pred))
      print(classification_report(y_test, pred))
      print("="*50)
  ```
### Results
Our analysis shows that incorporating sentiment analysis into churn prediction models can significantly improve their accuracy. Key findings include:

* Gradient Boosting (87%) and Random Forest (86.6%) demonstrated superior predictive accuracy and the ability to capture complex non-linear relationships in churn data.
* Logistic Regression (76%) provided a foundational linear analysis, while the Decision Tree model (68%) struggled with nuanced relationship capture.
 ## Key Predictors of Churn:

* Age and Balance were positively correlated with churn, indicating that older customers with higher balances are at higher risk.
* Active membership showed a negative correlation, emphasizing the importance of customer engagement in reducing churn.

## Challenges and Limitations

#### Model-Specific Constraints:

* Logistic Regression's linear assumptions limit its ability to capture complex patterns, while Decision Tree models struggle with nuanced relationships, leading to lower accuracy (68%).
 #### Data Quality and Feature Engineering:

* Insufficient or unrepresentative data may affect model performance, and the need for advanced feature engineering is critical to improving predictive accuracy and extracting meaningful insights.
#### Ethical and Practical Considerations:

* Bias in data or model predictions could lead to unfair treatment of specific customer groups. Implementing ethical AI frameworks and addressing fairness concerns remains a challenge.







## Recommendations and Future Scope:

* Employ tailored strategies for older and high-balance customers and enhance engagement beyond active membership.
* Explore hybrid machine learning models and real-time churn prediction systems to improve accuracy and timeliness.
* Focus on ethical AI and advanced data collection for robust, fair, and actionable churn management.

