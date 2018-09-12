
### Regularization Exercise

Perhaps it's not too surprising at this point, but there are classes in sklearn that will help you perform regularization with your linear regression. You'll get practice with implementing that in this exercise. In this assignment's data.csv, you'll find data for a bunch of points including six predictor variables and one outcome variable. Use sklearn's Lasso class to fit a linear regression model to the data, while also using L1 regularization to control for model complexity.

Perform the following steps below:
1. Load in the data

The data is in the file called 'data.csv'.
Split the data so that the six predictor features (first six columns) are stored in X, and the outcome feature (last column) is stored in y.
2. Fit data using linear regression with Lasso regularization

Create an instance of sklearn's Lasso class and assign it to the variable lasso_reg. You don't need to set any parameter values: use the default values for the quiz.
Use the Lasso object's .fit() method to fit the regression model onto the data.
3. Inspect the coefficients of the regression model

Obtain the coefficients of the fit regression model using the .coef_ attribute of the Lasso object. Store this in the reg_coef variable: the coefficients will be printed out, and you will use your observations to answer the question at the bottom of the page.


```python
from sklearn.linear_model import Lasso
from pandas import read_csv
from sklearn.metrics import mean_squared_error

#-------- read data ---------
train_data = read_csv('data.csv',header=None)
X = train_data.iloc[:,0:6]
y = train_data.iloc[:,-1]
#-----------------------------

#-------- fit model & predict---
lasso_reg = Lasso()
lasso_reg.fit(X, y)
y_pred1 = lasso_reg.predict(X)
#-----------------------------


print(lasso_reg.coef_)
print('mse:', mean_squared_error(y,y_pred1))
```

    [ 0.          2.35793224  2.00441646 -0.05511954 -3.92808318  0.        ]
    mse: 3.6640336296100657



```python
# fit in the regular model
from sklearn.linear_model import LinearRegression

#-------- fit model & predict---
model = LinearRegression()
model.fit(X, y)
y_pred2 = model.predict(X)
#-----------------------------

print(model.coef_)
print('mse:', mean_squared_error( y,y_pred2))
```

    [-6.19918532e-03  2.96325160e+00  1.98199191e+00 -7.86249920e-02
     -3.95818772e+00  9.30786141e+00]
    mse: 0.7266638169938208


### Feature Scaling Exercise

Previously, you saw how regularization will remove features from a model (by setting their coefficients to zero) if the penalty for removing them is small. In this exercise, you'll revisit the same dataset as before and see how scaling the features changes which features are favored in a regularization step. See the "Quiz: Regularization" page for more details. The only thing different for this quiz compared to the previous one is the addition of a new step after loading the data, where you will use sklearn's StandardScaler to standardize the data before you fit a linear regression model to the data with L1 (Lasso) regularization.


```python
from sklearn.preprocessing import StandardScaler

#-------- read data ---------
train_data = read_csv('data.csv',header=None)
X = train_data.iloc[:,0:6]
y = train_data.iloc[:,-1]
#-----------------------------


# Create the standardization scaling object.
scaler = StandardScaler()

# Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

#-------- fit model & predict---
lasso_reg = Lasso()
lasso_reg.fit(X_scaled, y)
y_pred1 = lasso_reg.predict(X_scaled)
#-----------------------------


print(lasso_reg.coef_)
print('mse:', mean_squared_error(y,y_pred1))
```

    [  0.           3.90753617   9.02575748  -0.         -11.78303187
       0.45340137]
    mse: 5.221618118385009

