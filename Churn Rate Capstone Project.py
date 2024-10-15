#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Load necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.linear_model import LogisticRegression  # For Logistic Regression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc  # For evaluation metrics
import statsmodels.api as sm  # For model summary

# 1. Reading the Data
# Replace 'file_path.csv' with the actual file path of your dataset
data = pd.read_csv("C:\\Users\\HP\\Downloads\\Customer Churn.csv")


# Check the structure of the dataset
data.info()

# 2. Exploratory Data Analysis (EDA)
# 2.1 Summary Statistics
print(data.describe())

# 2.2 Checking the Distribution of the Target Variable (Exited)
sns.countplot(x='Exited', data=data, palette='Blues')
plt.title('Distribution of Exited (Churn)')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()

# 2.3 Visualizing Numerical Variables
# Histogram for Age
sns.histplot(data['Age'], bins=10, color='lightgreen', kde=False, edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Balance Distribution
sns.histplot(data['Balance'], bins=20, color='lightblue', kde=False, edgecolor='black')
plt.title('Distribution of Balance')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

# Estimated Salary Distribution
sns.histplot(data['EstimatedSalary'], bins=20, color='coral', kde=False, edgecolor='black')
plt.title('Distribution of Estimated Salary')
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.show()

# 2.4 Categorical Variables
# Geography vs Exited
sns.countplot(x='Geography', hue='Exited', data=data, palette='Set1')
plt.title('Geography vs Exited (Churn)')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()

# Gender vs Exited
sns.countplot(x='Gender', hue='Exited', data=data, palette='Set2')
plt.title('Gender vs Exited (Churn)')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 2.5 Correlation Analysis
# Correlation matrix for numeric variables
numeric_vars = data[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']]
cor_matrix = numeric_vars.corr()

# Plotting correlation matrix
sns.heatmap(cor_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Numeric Variables')
plt.show()

# 3. Data Preprocessing
# Convert categorical variables to numeric using one-hot encoding or label encoding
data['Geography'] = data['Geography'].astype('category').cat.codes
data['Gender'] = data['Gender'].astype('category').cat.codes
data['HasCrCard'] = data['HasCrCard'].astype('category').cat.codes
data['IsActiveMember'] = data['IsActiveMember'].astype('category').cat.codes
data['Exited'] = data['Exited'].astype('category').cat.codes  # Convert target variable to numeric

# Check for any missing values
print(data.isna().sum())

# 4. Splitting the Data into Training and Testing Sets
X = data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = data['Exited']

# 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# 5. Building the Logistic Regression Model
log_reg = sm.Logit(y_train, sm.add_constant(X_train))  # Adding constant for intercept
model = log_reg.fit()

# Summary of the model to check the coefficients
print(model.summary())

# 6. Making Predictions
predictions_prob = model.predict(sm.add_constant(X_test))  # Get predicted probabilities
predictions = np.where(predictions_prob > 0.5, 1, 0)  # Convert probabilities to binary output (0 or 1)

# 7. Evaluating the Model
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print('Confusion Matrix:\n', conf_matrix)

# 8. Model Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy:.4f}')

# 9. ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, predictions_prob)
roc_auc = auc(fpr, tpr)

# Plotting the ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print(f'AUC Value: {roc_auc:.4f}')


# In[ ]:




