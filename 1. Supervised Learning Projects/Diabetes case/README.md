
### Diabetes Case Study

You now have had the opportunity to work with a range of supervised machine learning techniques for both classification and regression.  Before you apply these in the project, let's do one more example to see how the machine learning process works from beginning to end with another popular dataset.

We will start out by reading in the dataset and our necessary libraries.  You will then gain an understanding of how to optimize a number of models using grid searching as you work through the notebook. 


```python
# Import our libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
sns.set(style="ticks")

import check_file as ch

%matplotlib inline

# Read in our dataset
diabetes = pd.read_csv('diabetes.csv')

# Take a look at the first few rows of the dataset
diabetes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Because this course has been aimed at understanding machine learning techniques, we have largely ignored items related to parts of the data analysis process that come before building machine learning models - exploratory data analysis, feature engineering, data cleaning, and data wrangling.  

> **Step 1:** Let's do a few steps here.  Take a look at some of usual summary statistics calculated to accurately match the values to the appropriate key in the dictionary below. 


```python
# Cells for work
proportion_of_diabetes = diabetes[diabetes.Outcome==1].shape[0] / diabetes.shape[0]
proportion_of_diabetes
```




    0.3489583333333333




```python
diabetes.hist(figsize=(12,8))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x10b74e0b8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10bb4da20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10bb7c4e0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10bba2f60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10bbcea20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10bbcea58>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10bc24f60>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10bc51a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10bc824e0>]],
          dtype=object)




![png](output_4_1.png)



```python
# Possible keys for the dictionary
a = '0.65'
b = '0'
c = 'Age'
d = '0.35'
e = 'Glucose'
f = '0.5'
g = "More than zero"

# Fill in the dictionary with the correct values here
answers_one = {
    'The proportion of diabetes outcomes in the dataset': d, # add letter here,
    'The number of missing data points in the dataset': b,# add letter here,
    'A dataset with a symmetric distribution': e,# add letter here,
    'A dataset with a right-skewed distribution': c,# add letter here, 
    'This variable has the strongest correlation with the outcome': e # add letter here
}

# Just to check your answer, don't change this
ch.check_one(answers_one)
```

    Awesome! These all look great!


> **Step 2**: Since our dataset here is quite clean, we will jump straight into the machine learning.  Our goal here is to be able to predict cases of diabetes.  First, you need to identify the y vector and X matrix.  Then, the following code will divide your dataset into training and test data.   


```python
y = diabetes.Outcome.values.tolist()# Pull y column
X = diabetes.drop('Outcome', axis=1)# Pull X variable columns
pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

Now that you have a training and testing dataset, we need to create some models that and ultimately find the best of them.  However, unlike in earlier lessons, where we used the defaults, we can now tune these models to be the very best models they can be.

It can often be difficult (and extremely time consuming) to test all the possible hyperparameter combinations to find the best models.  Therefore, it is often useful to set up a randomized search.  

In practice, randomized searches across hyperparameters have shown to be more time confusing, while still optimizing quite well.  One article related to this topic is available [here](https://blog.h2o.ai/2016/06/hyperparameter-optimization-in-h2o-grid-search-random-search-and-the-future/).  The documentation for using randomized search in sklearn can be found [here](http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py) and [here](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

In order to use the randomized search effectively, you will want to have a pretty reasonable understanding of the distributions that best give a sense of your hyperparameters.  Understanding what values are possible for your hyperparameters will allow you to write a grid search that performs well (and doesn't break).

> **Step 3**: In this step, I will show you how to use randomized search, and then you can set up grid searches for the other models in Step 4.  However, you will be helping, as I don't remember exactly what each of the hyperparameters in SVMs do.  Match each hyperparameter to its corresponding tuning functionality.




```python
# build a classifier
clf_rf = RandomForestClassifier()

# Set up the hyperparameter search
param_dist = {"max_depth": [3, None],
              "n_estimators": list(range(10, 200)),
              "max_features": list(range(1, X_test.shape[1]+1)),
              "min_samples_split": list(range(2, 11)),
              "min_samples_leaf": list(range(1, 11)),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# Run a randomized search over the hyperparameters
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist)

# Fit the model on the training data
random_search.fit(X_train, y_train)

# Make predictions on the test data
rf_preds = random_search.best_estimator_.predict(X_test)

ch.print_metrics(y_test, rf_preds, 'random forest')
```

    Accuracy score for random forest : 0.7532467532467533
    Precision score random forest : 0.6440677966101694
    Recall score random forest : 0.6909090909090909
    F1 score random forest : 0.6666666666666665
    
    
    


> **Step 4**: Now that you have seen how to run a randomized grid search using random forest, try this out for the AdaBoost and SVC classifiers.  You might also decide to try out other classifiers that you saw earlier in the lesson to see what works best.


```python
# build a classifier for ada boost
clf_rf = AdaBoostClassifier()

# Set up the hyperparameter search
# look at  setting up your search for n_estimators, learning_rate
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
param_dist = {
    "learning_rate": [0.001, 0.005, 0.05, 0.1, 0.3, 0.4, 0.5],
    "n_estimators": list(range(10, 200))
}

# Run a randomized search over the hyperparameters
ada_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist)

# Fit the model on the training data
ada_search.fit(X_train, y_train)

# Make predictions on the test data
ada_preds = ada_search.best_estimator_.predict(X_test)

# Return your metrics on test data
ch.print_metrics(y_test, ada_preds, 'adaboost')
```

    Accuracy score for adaboost : 0.7727272727272727
    Precision score adaboost : 0.6923076923076923
    Recall score adaboost : 0.6545454545454545
    F1 score adaboost : 0.6728971962616823
    
    
    



```python
# build a classifier for ada boost
clf_rf = SVC()

# Set up the hyperparameter search
# look at  setting up your search for n_estimators, learning_rate
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
param_dist = {"C": [0.1, 0.5, 1, 3, 5],
              "kernel": ['linear','rbf']
             }

# Run a randomized search over the hyperparameters
svc_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist)

# Fit the model on the training data
svc_search.fit(X_train, y_train)

# Make predictions on the test data
svc_preds = svc_search.best_estimator_.predict(X_test)

# Return your metrics on test data
ch.print_metrics(y_test, svc_preds, 'svc')
```

    Accuracy score for svc : 0.7532467532467533
    Precision score svc : 0.6545454545454545
    Recall score svc : 0.6545454545454545
    F1 score svc : 0.6545454545454545
    
    
    


> **Step 5**: Use the test below to see if your best model matched, what we found after running the grid search.  


```python
a = 'randomforest'
b = 'adaboost'
c = 'supportvector'

best_model =  b# put your best model here as a string or variable

# See if your best model was also mine.  
# Notice these might not match depending your search!
ch.check_best(best_model)
```

    Nice!  It looks like your best model matches the best model I found as well!  It makes sense to use f1 score to determine best in this case given the imbalance of classes.  There might be justification for precision or recall being the best metric to use as well - precision showed to be best with adaboost again.  With recall, SVMs proved to be the best for our models.


Once you have found your best model, it is also important to understand why it is performing well.  In regression models where you can see the weights, it can be much easier to interpret results. 

> **Step 6**:  Despite the fact that your models here are more difficult to interpret, there are some ways to get an idea of which features are important.  Using the "best model" from the previous question, find the features that were most important in helping determine if an individual would have diabetes or not. Do your conclusions match what you might have expected during the exploratory phase of this notebook?


```python
# Show your work here - the plot below was helpful for me
# https://stackoverflow.com/questions/44101458/random-forest-feature-importance-chart-using-python
# load_diabetes

features = diabetes.columns

clfs = [(random_search, 'random forest'), (ada_search, 'adaBoost'), (svc_search, 'svm')]

for clf, title in clfs:
    importances = clf.best_estimator_.feature_importances_
    indices = np.argsort(importances)


    plt.title('{title} Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


```


    --------------------------------------------------------------------------

    NameError                                Traceback (most recent call last)

    <ipython-input-50-34e41105bfd1> in <module>()
          5 features = diabetes.columns
          6 
    ----> 7 clfs = [(random_search, 'random forest'), (ada_search, 'adaBoost'), (svc_search, 'svm')]
          8 
          9 for clf, title in clfs:


    NameError: name 'svc_search' is not defined


> **Step 7**:  Using your results above to complete the dictionary below.


```python
# Check your solution by matching the correct values in the dictionary
# and running this cell
a = 'Age'
b = 'BloodPressure'
c = 'BMI'
d = 'DiabetesPedigreeFunction'
e = 'Insulin'
f = 'Glucose'
g = 'Pregnancy'
h = 'SkinThickness'



sol_seven = {
    'The variable that is most related to the outcome of diabetes' : # letter here,
    'The second most related variable to the outcome of diabetes' : # letter here,
    'The third most related variable to the outcome of diabetes' : # letter here,
    'The fourth most related variable to the outcome of diabetes' : # letter here
}

ch.check_q_seven(sol_seven)
```

> **Step 8**:  Now provide a summary of what you did through this notebook, and how you might explain the results to a non-technical individual.  When you are done, check out the solution notebook by clicking the orange icon in the upper left.
