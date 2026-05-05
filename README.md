# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import NumPy, dataset loader, regression models, preprocessing tools, and evaluation metrics from scikit-learn.

2.Load the California housing dataset using the built-in dataset function.

3.Select the first three columns as independent variables and form two dependent variables using target and another feature column.

4.Split the dataset into training and testing sets using train-test split.

5.Apply standard scaling to the training and testing independent variables.

6.Apply standard scaling to the training and testing dependent variables.

7.Create an SGD regressor model with defined iteration and tolerance values.

8.Wrap the SGD regressor inside a multi-output regressor to handle multiple outputs.

9.Train the multi-output regression model using the scaled training data.

10.Predict the output values for the testing dataset.

11.Convert the predicted and actual values back to their original scale using inverse transformation.

12.Compute the mean squared error to evaluate the model performance.

13.Display the mean squared error and the first few predicted values.
## Program:
```

Developed by: Rasindhan R
RegisterNumber: 212224230222

```
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X=data.data[:,:3]
Y=np.column_stack((data.target,data.data[:,6]))
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train =scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd=SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Square Error:",mse)
print("\nPredictions:\n",Y_pred[:5])


```

## Output:
<img width="474" height="180" alt="Screenshot 2026-01-30 112746" src="https://github.com/user-attachments/assets/d62f3bb6-d4c0-44eb-aaf2-9fde9ba83735" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
