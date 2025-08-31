# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.

2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.

3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.

4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.

## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: NIVETHA S
RegisterNumber: 212223040137
```
 
``` Python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

```
<img width="1920" height="1080" alt="Screenshot 2025-08-28 081417" src="https://github.com/user-attachments/assets/5bd39e25-68c4-4544-99ef-b8daeea9db9d" />

```python
X=df.drop(columns=['AveOccup','HousingPrice'])
Y=df[['AveOccup','HousingPrice']]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print('Name: NIVETHA S')
print('Reg no: 212223040137')
print("Mean Squared Error:", mse)
print("\nPredictions: \n",Y_pred[:5])
```

## Output:

<img width="904" height="322" alt="Screenshot 2025-08-31 160350" src="https://github.com/user-attachments/assets/ba58965f-8007-4e10-9817-83b55a1cdcec" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
