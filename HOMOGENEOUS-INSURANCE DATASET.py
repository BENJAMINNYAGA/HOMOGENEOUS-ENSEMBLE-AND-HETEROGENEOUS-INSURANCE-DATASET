#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression 


# In[81]:


df=pd.read_csv("C:/Users/BENJAMIN/Downloads/insurance.csv")


# In[82]:


df.info()


# In[83]:


df.head()


# In[84]:


columns=df.columns.values.tolist()
print(columns)


# In[85]:


# Check for missing data
df.isnull().sum()

# Impute missing data with mean
df = df.fillna(df.mean())
print (df)


# In[86]:


from sklearn.preprocessing import LabelEncoder
cat_cols = ["sex", "smoker", "region"]
# Create LabelEncoder object
le = LabelEncoder()
# Encode the categorical features
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
# Print the encoded dataset
print(df)


# In[87]:


# Separate features and target
X = df.drop(['charges'], axis=1)
y = df['charges']


# In[88]:


# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Print the normalized data
print(X_norm)


# In[89]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[90]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
# Initialize random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model on training set
rf.fit(X_train, y_train)

# Make predictions on testing set
y_pred = rf.predict(X_test)
print(y_pred)


# In[92]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print evaluation metrics
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# In[71]:


from flask import Flask, jsonify, request
# Create a Flask app
app = Flask(__name__)

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the request
    features = request.json['features']
     # Convert the features to a DataFrame
    X = pd.DataFrame.from_dict(features)
     # Return the prediction as a JSON response
    response = {'prediction': y_pred.tolist()}
    return jsonify(response)
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    ''


# In[ ]:




