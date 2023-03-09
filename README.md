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
# Conclussion
This project demonstrates the use of a homogeneous ensemble random forest on the insurance.csv dataset. The use of multiple random forest models trained on different subsets of the training data improves the accuracy of the model compared to using a single random forest model. The final model achieves an accuracy of 86% on the testing set.
By analyzing the customer churn dataset using a heterogeneous ensemble of machine learning algorithms, with a focus on using the random forest algorithm, we aim to identify the factors that contribute to insurance cover and develop a predictive model to identify customers who are surposed to pay for insurance cover. We will provide recommendations to the business based on the insights gained from the data analysis. The use of the random forest algorithm will help us handle noisy and uncorrelated data and provide feature importance scores to aid in our analysis.
# Model deployment
After training and evaluating our homogeneous ensemble random forest model on the insurance dataset, we are now ready to deploy the model for use in a production environment. Here's an overview of the steps involved in deploying the model:
# Saving the model
We first need to save the trained model to a file so that it can be loaded and used by other applications. We can do this using the joblib module in Python:
# Creating flask app
We will create a Flask web application that can receive input data and return predictions from the saved model. Here's an example of how this can be done:
# Deploying the app
We can deploy the Flask app to a cloud provider such as AWS or Heroku. Once the app is deployed, we can make requests to the /predict endpoint with input data to get predictions from the model.
That's it! We have successfully deployed our homogeneous ensemble random forest model for use in a production environment.
# Reference
This project was completed with the help of the following resources:
[Churn Modelling Dataset on Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/insurance dataset)
[scikit-learn Documentation](https://scikit-learn.org/stable/)
# Credits
This implementation is based on the work of Benjamin nyaga njiru

