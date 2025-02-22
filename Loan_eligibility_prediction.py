# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:38:40 2025

@author: Mbekezeli Tshabalala
"""
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Load the csv file
dat = pd.read_csv('C:\\Users\\Mbekezeli Tshabalala\\OneDrive\\Documents\\My projects\\Hex Softwares\\Week 1\\Project_3\\Loan_Data.csv')

# Show the information of the data
dat.info()

# Show duplicate values
dat.duplicated().sum() # No duplicates

# Show missing values
dat.isna().sum()

# Handling missing values of categorical variables
dat['Gender'] = dat['Gender'].fillna(dat['Gender'].mode()[0])
dat['Married'] = dat['Married'].fillna(dat['Married'].mode()[0])
dat['Dependents'] = dat['Dependents'].fillna(dat['Dependents'].mode()[0])
dat['Self_Employed'] = dat['Self_Employed'].fillna(dat['Self_Employed'].mode()[0])
dat['Credit_History'] = dat['Credit_History'].fillna(dat['Credit_History'].mode()[0])


# Handling missing values of numeric variables
sns.boxplot(x='LoanAmount', data=dat)
dat['LoanAmount'].fillna(dat['LoanAmount'].median(),inplace=True) #Since the variable has outliers, I will impute with the median

sns.boxplot(x='Loan_Amount_Term', data=dat)
dat['Loan_Amount_Term'].fillna(dat['Loan_Amount_Term'].median(), inplace=True) #Since the variable has outliers, I will impute with the median

#Dropping the Loan_ID column
dat.drop(columns=['Loan_ID'], axis=1, inplace=True)

#Dealing with categorical data
cat_col = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
dat = pd.get_dummies(dat, columns= cat_col, drop_first=True) # Apply one-hot Encoding and avoids multicollinearity
cat_bool = dat.select_dtypes('bool').columns.tolist()
dat[cat_bool] = dat[cat_bool].astype(int)
dat['Loan_Status'] = dat['Loan_Status'].map({'Y': 1, 'N': 0})
# Normalizing numeric values
num_variables = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
scaler = MinMaxScaler()
dat[num_variables] = scaler.fit_transform(dat[num_variables])

# Separate the features from the target
target = dat['Loan_Status']
features = dat.drop(columns='Loan_Status',axis=1)

# split dataset into training ad testing sets
features_train,features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fitting a decision tree model
DTClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
DTClassifier.fit(features_train, target_train)

# Predicting
y_pred= DTClassifier.predict(features_test)

# Display the accuracy
print('The accuracy of decision tree is:', metrics.accuracy_score(y_pred, target_test))

# Fitting a Naive Bayes model
NBClassifier = GaussianNB()
NBClassifier.fit(features_train,target_train)
yhat = NBClassifier.predict(features_test)
print('The accuracy of Naive Bayes is:', metrics.accuracy_score(yhat, target_test))

"""
Interpreting the model
"""
# Based on the results of decision tree and Naive Bayes algorithms, the Naive Bayes algorithm is better
