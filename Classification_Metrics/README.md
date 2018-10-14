
### Our Mission

In this lesson you gained some insight into a number of techniques used to understand how well our model is performing.  This notebook is aimed at giving you some practice with the metrics specifically related to classification problems.  With that in mind, we will again be looking at the spam dataset from the earlier lessons.

First, run the cell below to prepare the data and instantiate a number of different models.
[solution](https://viewb23fd40d.udacity-student-workspaces.com/notebooks/Classification_Metrics_Solution.ipynb)


```python
# Import our libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import tests as t

# Read in our dataset
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Fix our response value
df['label'] = df.label.map({'ham':0, 'spam':1})

# Split our dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# Instantiate a number of our models
naive_bayes = MultinomialNB()
bag_mod = BaggingClassifier(n_estimators=200)
rf_mod = RandomForestClassifier(n_estimators=200)
ada_mod = AdaBoostClassifier(n_estimators=300, learning_rate=0.2)
svm_mod = SVC()
```

> **Step 1**: Now, fit each of the above models to the appropriate data.  Answer the following question to assure that you fit the models correctly.


```python
# Fit each of the 4 models
# This might take some time to run
naive_bayes.fit(training_data,y_train)
bag_mod.fit(training_data,y_train)
rf_mod.fit(training_data,y_train)
ada_mod.fit(training_data,y_train)
svm_mod.fit(training_data,y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
# # The models you fit above were fit on which data?

# a = 'X_train'
# b = 'X_test'
# c = 'y_train'
# d = 'y_test'
# e = 'training_data'
# f = 'testing_data'

# # Change models_fit_on to only contain the correct string names
# # of values that you oassed to the above models

# models_fit_on = f # update this to only contain correct letters

# # Checks your solution - don't change this
# t.test_one(models_fit_on)
```

> **Step 2**: Now make predictions for each of your models on the data that will allow you to understand how well our model will extend to new data.  Then correctly add the strings to the set in the following cell.


```python
# Make predictions using each of your models
preds_nb = naive_bayes.predict(testing_data)
bag_nb = bag_mod.predict(testing_data)
rf_nb = rf_mod.predict(testing_data)
ada_nb = ada_mod.predict(testing_data)
svm_nb = svm_mod.predict(testing_data)
```


```python
# # Which data was used in the predict method to see how well your
# # model would work on new data?

# a = 'X_train'
# b = 'X_test'
# c = 'y_train'
# d = 'y_test'
# e = 'training_data'
# f = 'testing_data'

# # Change models_predict_on to only contain the correct string names
# # of values that you oassed to the above models

# models_predict_on = {a, b, c, d, e, f} # update this to only contain correct letters

# # Checks your solution - don't change this
# t.test_two(models_predict_on)
```

Now that you have set up all your predictions, let's get to topis addressed in this lesson - measuring how well each of your models performed. First, we will focus on how each metric was calculated for a single model, and then in the final part of this notebook, you will choose models that are best based on a particular metric.

You will be writing functions to calculate a number of metrics and then comparing the values to what you get from sklearn.  This will help you build intuition for how each metric is calculated.

> **Step 3**: As an example of how this will work for the upcoming questions, run the cell below.  Fill in the below function to calculate accuracy, and then compare your answer to the built in to assure you are correct.


```python
# accuracy is the total correct divided by the total to predict
def accuracy(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the accuracy as a float
    '''
    return np.sum(preds == actual)/len(actual)


print(accuracy(y_test, preds_nb))
print(accuracy_score(y_test, preds_nb))
print("Since these match, we correctly calculated our metric!")
```

    0.9885139985642498
    0.9885139985642498
    Since these match, we correctly calculated our metric!


> **Step 4**: Fill in the below function to calculate precision, and then compare your answer to the built in to assure you are correct.


```python
# precision is the true positives over the predicted positive values
def precision(actual, preds):
    '''
    INPUT
    (assumes positive = 1 and negative = 0)
    preds - predictions as a numpy array or pandas series 
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the precision as a float
    '''
    a = len(actual) - np.sum(preds == 0) - np.sum(actual == 0)
    b = np.sum(preds == actual)
    true_positive = (a + b) / 2
    return true_positive / np.sum(preds == 1) # calculate precision here


print(precision(y_test, preds_nb))
print(precision_score(y_test, preds_nb))
print("If the above match, you got it!")
```

    0.9720670391061452
    0.9720670391061452
    If the above match, you got it!


> **Step 5**: Fill in the below function to calculate recall, and then compare your answer to the built in to assure you are correct.


```python
# recall is true positives over all actual positive values
def recall(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the recall as a float
    '''

    a = len(actual) - np.sum(preds == 0) - np.sum(actual == 0)
    b = np.sum(preds == actual)
    true_positive = (a + b) / 2
    return true_positive / np.sum(actual == 1) # calculate recall here


print(recall(y_test, preds_nb))
print(recall_score(y_test, preds_nb))
print("If the above match, you got it!")
```

    0.9405405405405406
    0.9405405405405406
    If the above match, you got it!


> **Step 6**: Fill in the below function to calculate f1-score, and then compare your answer to the built in to assure you are correct.


```python
# f1_score is 2*(precision*recall)/(precision+recall))
def f1(actual, preds):
    '''
    INPUT
    preds - predictions as a numpy array or pandas series
    actual - actual values as a numpy array or pandas series
    
    OUTPUT:
    returns the f1score as a float
    '''
    
#     return 2*(precision(y_test, preds)*recall(y_test, preds))/(precision(y_test, preds) + recall(y_test, preds)) # calculate f1-score here
    tp = len(np.intersect1d(np.where(preds==1), np.where(actual==1)))
    pred_pos = (preds==1).sum()
    prec = tp/(pred_pos)
    act_pos = (actual==1).sum()
    recall = tp/act_pos
    return 2*(prec*recall)/(prec+recall)


print(f1(y_test, preds_nb))
print(f1_score(y_test, preds_nb))
print("If the above match, you got it!")
```

    0.9560439560439562
    0.9560439560439562
    If the above match, you got it!


> **Step 7:** Now that you have calculated a number of different metrics, let's tie that to when we might use one versus another.  Use the dictionary below to match a metric to each statement that identifies when you would want to use that metric.


```python
# add the letter of the most appropriate metric to each statement
# in the dictionary
a = "recall"
b = "precision"
c = "accuracy"
d = 'f1-score'


seven_sol = {
'We have imbalanced classes, which metric do we definitely not want to use?': c, # letter here,
'We really want to make sure the positive cases are all caught even if that means we identify some negatives as positives': a, # letter here,    
'When we identify something as positive, we want to be sure it is truly positive': b, # letter here, 
'We care equally about identifying positive and negative cases': d, # letter here    
}

# t.sol_seven(seven_sol)
# c,a,b,d output
# That's right!  It isn't really necessary to memorize these in practice, 
# but it is important to know they exist and know why might use one metric 
# over another for a particular situation.
```

> **Step 8:** Given what you know about the metrics now, use this information to correctly match the appropriate model to when it would be best to use each in the dictionary below.


```python
# use the answers you found to the previous questiona, then match the model that did best for each metric
a = "naive-bayes"
b = "bagging"
c = "random-forest"
d = 'ada-boost'
e = "svm"


preds_dic = {'nb':preds_nb, 'bag': bag_nb, 'rf': rf_nb, 'ada': ada_nb, 'svm': svm_nb}
for name, pred in preds_dic.items():
    print(name, ' accuracy: ', accuracy(y_test, pred))
    print(name, ' precision: ', precision(y_test, pred))
    print(name, ' recall: ', recall(y_test, pred))
    print(name, ' f1: ', f1( y_test, pred))
eight_sol = {
'We have imbalanced classes, which metric do we definitely not want to use?': a, # letter here,
'We really want to make sure the positive cases are all caught even if that means we identify some negatives as positives': a, # letter here,    
'When we identify something as positive, we want to be sure it is truly positive': c, # letter here, 
'We care equally about identifying positive and negative cases': a, # letter here  
}

# t.sol_eight(eight_sol)
# output: That's right!  Naive Bayes was the best model for all of our metrics except precision!
```

    nb  accuracy:  0.9885139985642498
    nb  precision:  0.9720670391061452
    nb  recall:  0.9405405405405406
    nb  f1:  0.9560439560439562
    bag  accuracy:  0.9741564967695621
    bag  precision:  0.9116022099447514
    bag  recall:  0.8918918918918919
    bag  f1:  0.9016393442622951
    rf  accuracy:  0.9820531227566404
    rf  precision:  1.0
    rf  recall:  0.8648648648648649
    rf  f1:  0.927536231884058
    ada  accuracy:  0.9770279971284996
    ada  precision:  0.9693251533742331
    ada  recall:  0.8540540540540541
    ada  f1:  0.9080459770114943
    svm  accuracy:  0.8671931083991385
    svm  precision:  nan
    svm  recall:  0.0
    svm  f1:  nan


    /Users/yangshao/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:15: RuntimeWarning: invalid value encountered in double_scalars
      from ipykernel import kernelapp as app
    /Users/yangshao/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:15: RuntimeWarning: invalid value encountered in long_scalars
      from ipykernel import kernelapp as app



```python
# cells for work
```


```python
# If you get stuck, also notice there is a solution available by hitting the orange button in the top left
```


```python


```

As a final step in this workbook, let's take a look at the last three metrics you saw, f-beta scores, ROC curves, and AUC.

**For f-beta scores:** If you decide that you care more about precision, you should move beta closer to 0.  If you decide you care more about recall, you should move beta towards infinity. 

> **Step 9:** Using the fbeta_score works similar to most of the other metrics in sklearn, but you also need to set beta as your weighting between precision and recall.  Use the space below to show that you can use [fbeta in sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) to replicate your f1-score from above.  If in the future you want to use a different weighting, [this article](http://mlwiki.org/index.php/Precision_and_Recall) does an amazing job of explaining how you might adjust beta for different situations.


```python
# import fbeta_score
from sklearn.metrics import fbeta_score

# Show that you can produce the same f1_score results using fbeta_score
print(fbeta_score(y_test, preds_nb, beta=1))
print(f1( y_test, preds_nb) == fbeta_score(y_test, preds_nb, beta=1))


```

    0.9560439560439562
    True


> **Step 10:** Building ROC curves in python is a pretty involved process on your own.  I wrote the function below to assist with the process and make it easier for you to do so in the future as well.  Try it out using one of the other classifiers you created above to see how it compares to the random forest model below.

Run the cell below to build a ROC curve, and retrieve the AUC for the random forest model.


```python
# Function for calculating auc and roc

def build_roc_auc(model, X_train, X_test, y_train, y_test):
    '''
    INPUT:
    model - an sklearn instantiated model
    X_train - the training data
    y_train - the training response values (must be categorical)
    X_test - the test data
    y_test - the test response values (must be categorical)
    OUTPUT:
    auc - returns auc as a float
    prints the roc curve
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from scipy import interp
    
    y_preds = model.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(y_test)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_preds[:, 1])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_preds[:, 1].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()
    
    return roc_auc_score(y_test, np.round(y_preds[:, 1]))
    
    
# Finding roc and auc for the random forest model    
build_roc_auc(rf_mod, training_data, testing_data, y_train, y_test) 

# not same value every time? Why????
```


![png](output_26_0.png)





    0.9216216216216216




```python
# Your turn here - choose another classifier to see how it compares
build_roc_auc(naive_bayes, training_data, testing_data, y_train, y_test) 



```


![png](output_27_0.png)





    0.9682007338464294




```python

build_roc_auc(bag_mod, training_data, testing_data, y_train, y_test) 
```


![png](output_28_0.png)





    0.9420261320923573




```python
build_roc_auc(rf_mod, training_data, testing_data, y_train, y_test) 
```


![png](output_29_0.png)





    0.9216216216216216




```python
build_roc_auc(ada_mod, training_data, testing_data, y_train, y_test) 
```


![png](output_30_0.png)





    0.9249574906031861




```python
build_roc_auc(naive_bayes, training_data, testing_data, y_train, y_test) 
```


![png](output_31_0.png)





    0.9682007338464294




```python
build_roc_auc(svm_mod, training_data, testing_data, y_train, y_test) 
```


    -----------------------------------------------------------

    AttributeError            Traceback (most recent call last)

    <ipython-input-64-c4a0a01eee30> in <module>()
    ----> 1 build_roc_auc(svm_mod, training_data, testing_data, y_train, y_test)
    

    <ipython-input-55-15096702fef3> in build_roc_auc(model, X_train, X_test, y_train, y_test)
         19     from scipy import interp
         20 
    ---> 21     y_preds = model.fit(X_train, y_train).predict_proba(X_test)
         22     # Compute ROC curve and ROC area for each class
         23     fpr = dict()


    ~/anaconda3/lib/python3.6/site-packages/sklearn/svm/base.py in predict_proba(self)
        588         datasets.
        589         """
    --> 590         self._check_proba()
        591         return self._predict_proba
        592 


    ~/anaconda3/lib/python3.6/site-packages/sklearn/svm/base.py in _check_proba(self)
        555     def _check_proba(self):
        556         if not self.probability:
    --> 557             raise AttributeError("predict_proba is not available when "
        558                                  " probability=False")
        559         if self._impl not in ('c_svc', 'nu_svc'):


    AttributeError: predict_proba is not available when  probability=False

