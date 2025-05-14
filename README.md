# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.
2. Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.
3. Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.
4. Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: AMAN SINGH
RegisterNumber:  212224040020
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Load the California Housing dataset
dataset = fetch_california_housing ()
df=pd. DataFrame(dataset.data, columns=dataset.feature_names)
df[ 'HousingPrice']=dataset.target
print(df.head())

```

```python
# Use the first 3 features as inputs
X = df.drop(columns=['AveOccup', 'HousingPrice'])#data[:, :3] # Features: 'MedInc', 'HouseAge', 'AveRooms '
# Use 'MedHouseVal' and 'AveOccup' as output variables
Y = df[['AveOccup', 'HousingPrice']]#np. column_stack((data.target, data.data[:, 6])) # Targets: 'MedHouseVal', 'AveOccup'
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)


#no tranformation required since it is only used for verification
#Y_test = scaler_Y.transform(Y_test)
```
```python
# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
# Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)
# Train the model
multi_output_sgd.fit(X_train, Y_train)
# Predict on the test data
```

```python

Y_pred = multi_output_sgd.predict(X_test)


# Inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)

#Y_test = scaler_Y.inverse_transform(Y_test)  #not required since Y_test didm't transform

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print ("Mean Squared Error:", mse,"\n\n")
#Optionally, print some predictions 
#print("\nPredictions: \n", Y_pred[:5]) # Print first 5 predictions

print("Actual test values:\n",Y_test)
print("Predicted test value\n",Y_pred)
```

## Output:
<img width="1068" alt="Screenshot 2025-04-21 at 8 53 10 AM" src="https://github.com/user-attachments/assets/2bb58b18-8a6d-4415-8b58-875a13a9070c" />
<img width="1036" alt="Screenshot 2025-04-21 at 8 53 46 AM" src="https://github.com/user-attachments/assets/2c388d82-7d4a-41ee-abcb-828ca371e922" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
