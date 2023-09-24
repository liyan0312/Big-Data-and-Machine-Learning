# -*- coding: utf-8 -*-

# 01-BU

"""

Describe the business objectives here

"""

# 02-DU

# Load Dataset
import pandas as pd
file = 'BankChurners_1.xls'
df= pd.read_excel(file)
print(df)
df.info()


# Explore Data
df.describe()
df_desc = df.describe()

df.corr()
df_corr=round(df.corr(),2)


# Add Visualisations

import seaborn as sns

df['Gender'].value_counts().plot.bar()
sns.distplot(df['Customer_Age'])
df['Dependent_count'].value_counts().sort_index().plot.bar()
df['Card_Category'].value_counts().sort_index().plot.bar()
df['Marital_Status'].value_counts().sort_index().plot.bar()
df['Income_Category'].value_counts().sort_index().plot.bar()
df['Education_Level'].value_counts().sort_index().plot.bar()




from collections import Counter
import matplotlib.pyplot as plt
plt.pie(Counter(df['Education_Level']).values(), labels=Counter(df['Education_Level']).keys(), autopct='%1.1f%%')


plt.pie(Counter(df['Attrition_Flag']).values(), labels=Counter(df['Attrition_Flag']).keys(), autopct='%1.1f%%')


import seaborn as sns
sns.countplot(x='Attrition_Flag', hue='Gender', data=df)


df.info()
df['Avg_Utilization_Ratio'].fillna(df['Avg_Utilization_Ratio'].median(), inplace=True)
df.info()

# Check for duplicate rows
is_duplicate = df.duplicated()
print(is_duplicate)

# Check for invailed data 'aa'
import numpy as np
invalid_rows = df[df['Total_Trans_Ct'] == 'aa']
df['Total_Trans_Ct'].describe()
df['Total_Trans_Ct'] = df['Total_Trans_Ct'].replace('aa', np.nan)

median_value = df['Total_Trans_Ct'].median()
df['Total_Trans_Ct'].fillna(median_value, inplace=True)



# 03-DP

# Add any pre-processing steps
# Deletion of useless data
df = df.drop(["CLIENTNUM","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)
df.head()
df.info()

# Change the date of Attrition_Flag to 0/1

df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

df.info()


df['Attrition_Flag'].describe()

df['Attrition_Flag'].value_counts().plot.pie()
df.corr()
df_corr=round(df.corr(),2)

df.info()
# Gendar issue
df['Gender'] = df['Gender'].replace({'Male': 'M'})
df['Gender'].value_counts().plot.bar()

df['Gender'].describe()




df['Gender'] = df.Gender.replace({'F':1,'M':0})

# one-Hot encoding 
df = pd.get_dummies(df, columns=["Education_Level", "Marital_Status", "Income_Category", "Card_Category"])
df = df.drop(columns=["Education_Level_Unknown", "Marital_Status_Unknown", "Income_Category_Unknown"])


df.corr()
df_corr=round(df.corr(),2)

# Data cleaning (converting 'Total Trans Ct' to an integer)
df["Total_Trans_Ct"] = df["Total_Trans_Ct"].astype(int)
df.info()

import pandas as pd
age_bins = [25, 35, 45, 55, 65, float('inf')]  
age_labels = ['26-35', '36-45', '46-55', '56-65', '66+']
df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, labels=age_labels)

print(df[['Customer_Age', 'Age_Group']])

df['Months_on_book'].describe()
book_bins = [12, 24, 36, 48, float('inf')]  
book_labels = ['1-2year', '2-3year', '3-4year', '4year+']
df['Register_Group'] = pd.cut(df['Months_on_book'], bins=book_bins, labels=book_labels)
print(df[['Register_Group', 'Months_on_book']])
df.info()
print(df[['Customer_Age', 'Age_Group']])
df['Age_Group'].value_counts().sort_index().plot.bar()
df['Register_Group'].value_counts().sort_index().plot.bar()


data_source1 = pd.read_excel('BankChurners_1.xls')
data_source2 = pd.read_excel('BankChurners_2.xlsx')
integrated_data = pd.merge(data_source1, data_source2, on='CLIENTNUM', how='inner')
integrated_data.to_csv('integrated_data.csv', index=False)


df.info()
df['Age_Group'] = df['Age_Group'].cat.codes
df['Register_Group'] = df['Register_Group'].cat.codes
df['Age_Group'].value_counts().plot.bar()
df['Register_Group'].value_counts().plot.bar()
df.info()


# 04-DT

X = df.drop(columns=['Attrition_Flag'])  # features
y = df['Attrition_Flag']  # Target variable

# Calculate the Pearson correlation coefficient between features and target variables
correlations = []
for feature in X.columns:
    corr = np.corrcoef(X[feature], y)[0, 1]
    correlations.append((feature, corr))

# Sort correlation results in descending absolute order
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# Print correlation results
for feature, corr in correlations:
    print(f"{feature}: {corr}")


threshold = 0.01
selected_features = [feature for feature, corr in correlations if abs(corr) >= threshold]
X_filtered = X[selected_features]
X_filtered.info()
print(X_filtered.head())


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Separating features and target variables
X = df.drop(columns=['Attrition_Flag'])  # features
y = df['Attrition_Flag']  # Target variable

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# Draw a chart of the proportion of cumulative explanatory variance

plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
# Using PCA for dimensionality reduction
n_components = 20  # Choose to retain 20 principal components
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)

df.info()
print("Explained Variance Ratios:", explained_variance_ratio[:n_components])

cumulative_variance_ratio = explained_variance_ratio.cumsum()

# data projection
# Create a PCA model and select dimensions after dimensionality reduction
n_projection_components = 2  # Select the number of dimensions to reduce to
pca_projection = PCA(n_components=n_projection_components)

# Using PCA model to fit merged projection dimensionality reduced data
X_projected = pca_projection.fit_transform(X_reduced)

df.info()


# 05-DMM

"""
Identify the Data Mining method
Describe how it aligns with the objectives

"""

# 06-DMA

#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# X represents the feature, y represents the target variable
X = X_filtered
y = df['Attrition_Flag']

# Split the data into a training set (70%) and a testing set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#Fit the model on the training set
rf_classifier.fit(X_train, y_train)

# #Label for Predictive Test Set
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation indicators
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)



#cart
from sklearn.tree import DecisionTreeClassifier

# Creating a CART classifier
cart_classifier = DecisionTreeClassifier(random_state=42)

#Fit the model on the training set
cart_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Label for Predictive Test Set
y_pred = cart_classifier.predict(X_test)

# Calculate evaluation indicators
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

#C5.0
from C50 import C50

# Assuming X_ Train and y_ Train contains training data
model = C50.C5_0(x=X_train, y=y_train)
# Assuming X_ Test contains test data
y_pred = model.predict(X_test)

# Calculate evaluation indicators
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)




from sklearn.linear_model import LogisticRegression

# Creating a binary logistic regression classifier
logistic_regression = LogisticRegression()

#Fit the model on the training set
logistic_regression.fit(X_train, y_train)

# Assuming X_ Test contains test data
y_pred = logistic_regression.predict(X_test)

# Calculate evaluation indicators
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)


import xgboost as xgb


#Create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)

#Fit the model on the training set
xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)

# Calculate evaluation indicators
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

feature_importances = xgb_classifier.feature_importances_



#Parameter tuning

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Creating a Random Forest Model
rf_classifier = RandomForestClassifier()

# Define the parameter range to search for
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Explicitly set to 'sqrt' or 'log2'
}

# Create a GridSearchCV object

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)

# Conduct parameter search on training data
grid_search.fit(X_train, y_train)

# Print optimal parameters
print("Best Parameters: ", grid_search.best_params_)

# Using the best parameter model for prediction
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)



# 07-DM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = X_filtered
y = df['Attrition_Flag']

# Divide training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, random_state=42)

# Fit the model on the training set
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#Predictive Test Set
y_pred = rf_classifier.predict(X_test)

# Calculate performance indicators
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

# Importance of acquiring features
# Create a DataFrame to correspond feature names with corresponding importance values
feature_importance = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Sort features in descending order of importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print Top 10 Features
top_n = 10  # 
print(f'Top {top_n} Important Features:')
print(feature_importance_df.head(top_n))


# Using a trained random forest model for prediction
y_pred = rf_classifier.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Calculate performance indicators
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)


# Statistical model prediction results
from collections import Counter

predictions_count = Counter(y_pred)
print("Predictions Count:", predictions_count)



import matplotlib.pyplot as plt
import seaborn as sns

#Importance of Visual Features
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

#Creating a DataFrame of Feature Importance
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

print("Feature Importance:\n", feature_importance_df)

#Draw a feature importance map
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 08-INT

import matplotlib.pyplot as plt
import seaborn as sns

#Importance of Visual Features
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

#Creating a DataFrame of Feature Importance
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

print("Feature Importance:\n", feature_importance_df)

#Draw a feature importance map
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()




X = X_filtered
y = df['Attrition_Flag']

# Divide training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, random_state=42)

# Fit the model on the training set
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import roc_curve, auc

# Predicted Probability
y_prob = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Draw ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()



def plot_learning_curve(estimator, title, X, y, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Draw a learning curve
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

from sklearn.model_selection import learning_curve

# Create a random forest classifier, and you can also set parameters as needed
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, random_state=42)

# Draw a learning curve
plot_learning_curve(rf_classifier, "Random Forest Learning Curve", X, y, cv=5)
plt.show()





#8.5


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = X_filtered
y = df['Attrition_Flag']

# Divide training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=400, max_depth=30, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, random_state=42)

# Fit the model on the training set
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#Predictive Test Set
y_pred = rf_classifier.predict(X_test)

# Calculate performance indicators
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# print result
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", conf_matrix)

# Importance of acquiring features
# Create a DataFrame to correspond feature names with corresponding importance values
feature_importance = rf_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

# Sort features in descending order of importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print Top 10 Features
top_n = 10  # 
print(f'Top {top_n} Important Features:')
print(feature_importance_df.head(top_n))
# Summarise Results

# Add relevant tables or graphs

# 09-ACT

"""

Desribe the Action Plan to Implement, Observe and Improve

"""
