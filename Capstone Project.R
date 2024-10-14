# Load necessary libraries
library(caret)  # For machine learning algorithms and model evaluation
library(dplyr)  # For data manipulation
library(ggplot2)  # For data visualization
library(corrplot)  # For correlation plot
library(pROC)  # For ROC and AUC

# 1. Reading the Data
# Replace 'file_path.csv' with the actual file path of your dataset
data <- read.csv(file.choose(),header = TRUE)

# Check the structure of the dataset
str(data)

# 2. Exploratory Data Analysis (EDA)
# 2.1 Summary Statistics
summary(data)

# 2.2 Checking the Distribution of the Target Variable (Exited)
ggplot(data, aes(x = Exited)) + 
  geom_bar(fill = 'steelblue') + 
  labs(title = 'Distribution of Exited (Churn)', x = 'Exited', y = 'Count')

# 2.3 Visualizing Numerical Variables
# Histogram for Age
ggplot(data, aes(x = Age)) + 
  geom_histogram(binwidth = 5, fill = 'lightgreen', color = 'black') + 
  labs(title = 'Distribution of Age', x = 'Age', y = 'Frequency')

# Balance Distribution
ggplot(data, aes(x = Balance)) + 
  geom_histogram(binwidth = 5000, fill = 'lightblue', color = 'black') + 
  labs(title = 'Distribution of Balance', x = 'Balance', y = 'Frequency')

# Estimated Salary Distribution
ggplot(data, aes(x = EstimatedSalary)) + 
  geom_histogram(binwidth = 10000, fill = 'coral', color = 'black') + 
  labs(title = 'Distribution of Estimated Salary', x = 'Estimated Salary', y = 'Frequency')

# 2.4 Categorical Variables
# Geography vs Exited
ggplot(data, aes(x = Geography, fill = Exited)) + 
  geom_bar(position = "dodge") + 
  labs(title = 'Geography vs Exited (Churn)', x = 'Geography', y = 'Count')

# Gender vs Exited
ggplot(data, aes(x = Gender, fill = Exited)) + 
  geom_bar(position = "dodge") + 
  labs(title = 'Gender vs Exited (Churn)', x = 'Gender', y = 'Count')

# 2.5 Correlation Analysis
# Correlation matrix for numeric variables
numeric_vars <- data %>% select(CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary)
cor_matrix <- cor(numeric_vars)
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.cex = 0.8)

# 3. Data Preprocessing
# Convert categorical variables to factors
data$Geography <- as.factor(data$Geography)
data$Gender <- as.factor(data$Gender)
data$HasCrCard <- as.factor(data$HasCrCard)
data$IsActiveMember <- as.factor(data$IsActiveMember)
data$Exited <- as.factor(data$Exited)  # Convert target variable to factor

# Check for any missing values
sum(is.na(data))

# 4. Splitting the Data into Training and Testing Sets
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(data$Exited, p = 0.7, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# 5. Building the Logistic Regression Model
model <- glm(Exited ~ CreditScore + Geography + Gender + Age + Tenure + Balance + 
               NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary, 
             family = binomial(link = "logit"), data = train_data)

# Summary of the model to check the coefficients
summary(model)

# 6. Making Predictions
predictions_prob <- predict(model, test_data, type = "response")  # Get predicted probabilities
predictions <- ifelse(predictions_prob > 0.5, 1, 0)  # Convert probabilities to binary output (0 or 1)

# 7. Evaluating the Model
# Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(predictions), test_data$Exited)
print(conf_matrix)

# 8. Model Accuracy
accuracy <- conf_matrix$overall['Accuracy']
print(paste("Model Accuracy:", round(accuracy, 4)))

# 9. ROC Curve and AUC
roc_curve <- roc(test_data$Exited, predictions_prob)
plot(roc_curve, col = "blue", main = "ROC Curve")
auc_value <- auc(roc_curve)
print(paste("AUC Value:", round(auc_value, 4)))
