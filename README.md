# HOMOGENEOUS-ENSEMBLE-AND-HETEROGENEOUS-INSURANCE-DATASET
Predictive analysis on insurance dataset using homogeneous and heterogeneous ensembles
# Introduction 
This project implements a homogeneous ensemble random forest algorithm on an insurance dataset. The goal is to predict whether a customer will file an insurance claim in the future based on various factors such as age, gender, location, and insurance history.
# Project dependencies
* Python 3.x
* numpy
* pandas
* scikit-learn
# Dataset
The dataset has to be downloaded from kaggle 
# Data preprocessing
Before training the model, the insurance dataset needs to be preprocessed. The preprocessing steps include:
Handling missing values: Replace missing values with the mean, median, or mode of the column or drop the rows/columns with missing values.
Encoding categorical variables: Convert categorical variables to numerical values using one-hot encoding or label encoding.
Feature scaling: Scale the features to a similar range to prevent any one feature from dominating the others.
# Model training
This will train the model using the RandomForestClassifier class from the scikit-learn package.
# Results
The accuracy of the homogeneous ensemble random forest model on the test set is 0.642. This is an improvement over a single decision tree, which has an accuracy of 0.619. The confusion matrix and classification report for the model can be found in the results folder.
# Conclusion
