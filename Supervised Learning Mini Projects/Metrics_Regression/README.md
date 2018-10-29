
### Boston Housing Data

In order to gain a better understanding of the metrics used in regression settings, we will be looking at the Boston Housing dataset.  

First use the cell below to read in the dataset and set up the training and testing data that will be used for the rest of this problem.


```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import tests2 as t

boston = load_boston()
y = boston.target
X = boston.data

X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
```

> **Step 1:** Before we get too far, let's do a quick check of the models that you can use in this situation given that you are working on a regression problem.  Use the dictionary and corresponding letters below to provide all the possible models you might choose to use.


```python
# When can you use the model - use each option as many times as necessary
a = 'regression'
b = 'classification'
c = 'both regression and classification'

models = {
    'decision trees': c,# Letter here,
    'random forest': c,# Letter here,
    'adaptive boosting':c, # Letter here,
    'logistic regression':b, # Letter here,
    'linear regression': a# Letter here
}

#checks your answer, no need to change this code
t.q1_check(models)
# output:
# That's right!  All but logistic regression can be used for predicting numeric values.  
# And linear regression is the only one of these that you should not use for predicting categories.  
# Technically sklearn won't stop you from doing most of anything you want, 
# but you probably want to treat cases in the way you found by answering this question!
```

    That's right!  All but logistic regression can be used for predicting numeric values.  And linear regression is the only one of these that you should not use for predicting categories.  Technically sklearn won't stop you from doing most of anything you want, but you probably want to treat cases in the way you found by answering this question!


> **Step 2:** Now for each of the models you found in the previous question that can be used for regression problems, import them using sklearn.


```python
# Import models from sklearn - notice you will want to use 
# the regressor version (not classifier) - googling to find 
# each of these is what we all do!
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
```

> **Step 3:** Now that you have imported the 4 models that can be used for regression problems, instantate each below.


```python
# Instantiate each of the models you imported
# For now use the defaults for all the hyperparameters
dt_model = DecisionTreeRegressor()
rf_model = RandomForestRegressor(n_estimators=200)
ada_model = AdaBoostRegressor(n_estimators=300, learning_rate=0.2)
lr_model = LinearRegression()
```

> **Step 4:** Fit each of your instantiated models on the training data.


```python
# Fit each of your models using the training data
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
ada_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



> **Step 5:** Use each of your models to predict on the test data.


```python
# Predict on the test values for each model
dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
ada_preds = ada_model.predict(X_test)
lr_preds = lr_model.predict(X_test)
```

> **Step 6:** Now for the information related to this lesson.  Use the dictionary to match the metrics that are used for regression and those that are for classification.
# potential model options
a = 'regression'
b = 'classification'
c = 'both regression and classification'

metrics = {
    'precision': b,# Letter here,
    'recall': b,# Letter here,
    'accuracy': b,# Letter here,
    'r2_score': a,# Letter here,
    'mean_squared_error': a,# Letter here,
    'area_under_curve': b,# Letter here, 
    'mean_absolute_area': a# Letter here 
}

#checks your answer, no need to change this code
t.q6_check(metrics)
> **Step 6:** Now that you have identified the metrics that can be used in for regression problems, use sklearn to import them.


```python
# Import the metrics from sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score,mean_squared_error, mean_absolute_error, roc_auc_score
```

> **Step 7:** Similar to what you did with classification models, let's make sure you are comfortable with how exactly each of these metrics is being calculated.  We can then match the value to what sklearn provides.


```python
def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual-preds)**2)
    sst = np.sum((actual-np.mean(actual))**2)
    return 1 - sse/sst

# Check solution matches sklearn
print(r2(y_test, dt_preds))
print(r2_score(y_test, dt_preds))
print("Since the above match, we can see that we have correctly calculated the r2 value.")
```

    0.7501287928872782
    0.7501287928872782
    Since the above match, we can see that we have correctly calculated the r2 value.


> **Step 8:** Your turn fill in the function below and see if your result matches the built in for mean_squared_error. 


```python
def mse(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean squared error as a float
    '''
    
    return np.sum((actual-preds)**2) / len(actual) # calculate mse here


# Check your solution matches sklearn
print(mse(y_test, dt_preds))
print(mean_squared_error(y_test, dt_preds))
print("If the above match, you are all set!")
```

    18.90988023952096
    18.90988023952096
    If the above match, you are all set!


> **Step 9:** Now one last time - complete the function related to mean absolute error.  Then check your function against the sklearn metric to assure they match. 


```python
def mae(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean absolute error as a float
    '''
    
    return np.sum(abs(actual-preds)) / len(actual) # calculate the mae here

# Check your solution matches sklearn
print(mae(y_test, dt_preds))
print(mean_absolute_error(y_test, dt_preds))
print("If the above match, you are all set!")
```

    3.0508982035928147
    3.0508982035928147
    If the above match, you are all set!


> **Step 10:** Which model performed the best in terms of each of the metrics?  Note that r2 and mse will always match, but the mae may give a different best model.  Use the dictionary and space below to match the best model via each metric.


```python
#match each metric to the model that performed best on it
a = 'decision tree'
b = 'random forest'
c = 'adaptive boosting'
d = 'linear regression'

def print_score(actual, preds, name):
    print('mse for ', name, 'is: ',mse(actual, preds))
    print('r2 for ', name, 'is: ',r2(actual, preds))    
    print('mae for ', name, 'is: ',mae(actual, preds), '\n')

print_score(y_test, dt_preds, a)
print_score(y_test, rf_preds, b )
print_score(y_test, ada_preds, c)
print_score(y_test, lr_preds, d)

best_fit = {
    'mse': b,# letter here,
    'r2': b,# letter here,
    'mae': b# letter here
}

#Tests your answer - don't change this code
t.check_ten(best_fit)
```

    mse for  decision tree is:  18.90988023952096
    r2 for  decision tree is:  0.7501287928872782
    mae for  decision tree is:  3.0508982035928147 
    
    mse for  random forest is:  10.675734110778437
    r2 for  random forest is:  0.8589330796765403
    mae for  random forest is:  2.166461077844309 
    
    mse for  adaptive boosting is:  15.239955774029408
    r2 for  adaptive boosting is:  0.79862240810798
    mae for  adaptive boosting is:  2.75242850833652 
    
    mse for  linear regression is:  20.747143360308847
    r2 for  linear regression is:  0.7258515818230061
    mae for  linear regression is:  3.15128783658839 
    
    That's right!  The random forest was best in terms of all the metrics this time!



```python
# cells for work
```
