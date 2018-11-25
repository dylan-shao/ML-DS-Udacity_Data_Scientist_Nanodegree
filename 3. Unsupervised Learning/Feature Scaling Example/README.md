
### Feature Scaling Example

You have now seen how feature scaling might change the clusters we obtain from the kmeans algorithm, but it is time to try it out!

First let's get some data to work with.  The first cell here will read in the necessary libraries, generate data, and make a plot of the data you will be working with throughout the rest of the notebook.

The dataset you will work with through the notebook is then stored in **data**.  


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from IPython.display import Image
from sklearn.datasets.samples_generator import make_blobs
import tests2 as t


%matplotlib inline

# DSND colors: UBlue, Salmon, Gold, Slate
plot_colors = ['#02b3e4', '#ee2e76', '#ffb613', '#2e3d49']

# Light colors: Blue light, Salmon light
plot_lcolors = ['#88d0f3', '#ed8ca1', '#fdd270']

# Gray/bg colors: Slate Dark, Gray, Silver
plot_grays = ['#1c262f', '#aebfd1', '#fafbfc']


def create_data():
    n_points = 120
    X = np.random.RandomState(3200000).uniform(-3, 3, [n_points, 2])
    X_abs = np.absolute(X)

    inner_ring_flag = np.logical_and(X_abs[:,0] < 1.2, X_abs[:,1] < 1.2)
    outer_ring_flag = X_abs.sum(axis = 1) > 5.3
    keep = np.logical_not(np.logical_or(inner_ring_flag, outer_ring_flag))

    X = X[keep]
    X = X[:60] # only keep first 100
    X1 = np.matmul(X, np.array([[2.5, 0], [0, 100]])) + np.array([22.5, 500])
    
    
    plt.figure(figsize = [6,6])

    plt.scatter(X1[:,0], X1[:,1], s = 64, c = plot_colors[-1])

    plt.xlabel('5k Completion Time (min)', size = 30)
    plt.xticks(np.arange(15, 30+5, 5), fontsize = 30)
    plt.ylabel('Test Score (raw)', size = 30)
    plt.yticks(np.arange(200, 800+200, 200), fontsize = 30)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [side.set_linewidth(2) for side in ax.spines.values()]
    ax.tick_params(width = 2)
    plt.savefig('C18_FeatScalingEx_01.png', transparent = True)
    
    
    data = pd.DataFrame(X1)
    data.columns = ['5k_Time', 'Raw_Test_Score']
    
    return data

data = create_data()
```


![png](output_1_0.png)


`1.` Take a look at the dataset.  Are there any missing values?  What is the average completion time?  What is the average raw test score?  Use the cells below to find the answers to these questions, and the dictioonary to match values and check against our solution.


```python
# cell for work
data.shape

mean_teset_score = np.mean(data['Raw_Test_Score'])

mean_tk_time = np.mean(data['5k_Time'])

print(mean_teset_score)
print(mean_tk_time)
```

    511.6996030976626
    22.899026685796343



```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 60 entries, 0 to 59
    Data columns (total 2 columns):
    5k_Time           60 non-null float64
    Raw_Test_Score    60 non-null float64
    dtypes: float64(2)
    memory usage: 1.0 KB



```python
data.describe()
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
      <th>5k_Time</th>
      <th>Raw_Test_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>60.000000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>22.899027</td>
      <td>511.699603</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.536244</td>
      <td>183.222427</td>
    </tr>
    <tr>
      <th>min</th>
      <td>15.263902</td>
      <td>206.597283</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.820638</td>
      <td>361.798208</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.033613</td>
      <td>545.795365</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.956643</td>
      <td>656.346547</td>
    </tr>
    <tr>
      <th>max</th>
      <td>29.867819</td>
      <td>797.599192</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Use the dictionary to match the values to the corresponding statements
a = 0
b = 60
c = 22.9
d = 4.53
e = 511.7

q1_dict = {
'number of missing values': a,# letter here,
'the mean 5k time in minutes': c,# letter here,    
'the mean test score as a raw value': e,# letter here,
'number of individuals in the dataset': b# letter here
}

# check your answer against ours here
t.check_q1(q1_dict)
```

    That looks right!


`2.` Now, instantiate a kmeans `model` with 2 cluster centers.  Use your model to `fit` and `predict` the the group of each point in your dataset.  Store the predictions in `preds`.  If you correctly created the model and predictions, you should see a top (blue) cluster and bottom (pink) cluster when running the following cell.


```python
model = KMeans(2)# instantiate a model with two centers
preds = model.fit(data).predict(data)# fit and predict
```


```python
# Run this to see your results

def plot_clusters(data, preds, n_clusters):
    plt.figure(figsize = [6,6])

    for k, col in zip(range(n_clusters), plot_colors[:n_clusters]):
        my_members = (preds == k)
        plt.scatter(data['5k_Time'][my_members], data['Raw_Test_Score'][my_members], s = 64, c = col)

    plt.xlabel('5k Completion Time (min)', size = 30)
    plt.xticks(np.arange(15, 30+5, 5), fontsize = 30)
    plt.ylabel('Test Score (raw)', size = 30)
    plt.yticks(np.arange(200, 800+200, 200), fontsize = 30)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [side.set_linewidth(2) for side in ax.spines.values()]
    ax.tick_params(width = 2)
    
plot_clusters(data, preds, 2)
```


![png](output_9_0.png)


`3.` Now create two new columns to add to your `data` dataframe.  The first is `test_scaled`, which you should create by subtracting the mean test score and dividing by the standard deviation test score.  

The second column to create is `5k_time_sec`, which should have the minutes changed to seconds.


```python
# your work here
data['test_scaled'] = (data['Raw_Test_Score'] - mean_teset_score) / np.std(data['Raw_Test_Score'])# standardized test scores
data['5k_time_sec'] = data['5k_Time'] * 60 # times in seconds
```

`4.` Now, similar to what you did in question 2, instantiate a kmeans `model` with 2 cluster centers.  Use your model to `fit` and `predict` the the group of each point in your dataset.  Store the predictions in `preds`.  If you correctly created the model and predictions, you should see a right (blue) cluster and left (pink) cluster when running the following cell.


```python
scaled_data = data[['5k_time_sec','test_scaled']]
```


```python
model = KMeans(2)# instantiate a model with two centers
preds = model.fit(scaled_data).predict(scaled_data)# fit and predict
```


```python
def plot_clusters2(data, preds, n_clusters):
    plt.figure(figsize = [6,6])

    for k, col in zip(range(n_clusters), plot_colors[:n_clusters]):
        my_members = (preds == k)
        plt.scatter(data.iloc[:,0][my_members], data.iloc[:,1][my_members], s = 64, c = col)

    plt.xlabel('5k Completion Time (s)', size = 15)
    x_min = np.min(data.iloc[:,0]) - 1
    x_max = np.max(data.iloc[:,0])+ 1
    plt.xticks(np.arange(x_min, x_max, (x_max-x_min)//4), fontsize = 15)
    
    plt.ylabel('Test Score', size = 15)
    y_min = np.min(data.iloc[:,1]) - 1
    y_max = np.max(data.iloc[:,1])+ 1
    plt.yticks(np.arange(y_min, y_max,(y_max-y_min)//5) , fontsize = 15)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    [side.set_linewidth(2) for side in ax.spines.values()]
    ax.tick_params(width = 2)
```


```python
# Run this to see your results
plot_clusters2(scaled_data, preds, 2)
```


![png](output_16_0.png)


`5.` Match the variable that best describes the way you should think of feature scaling with algorithms that use distance based metrics or regularization.


```python
# options
a = 'We should always use normalizing'
b = 'We should always scale our variables between 0 and 1.'
c = 'Variable scale will frequently influence your results, so it is important to standardize for all of these algorithms.'
d = 'Scaling will not change the results of your output.'

best_option = c# best answer variable here


# check your answer against ours here
t.check_q5(best_option)
```




    <IPython.core.display.Image object>



###  If you get stuck, you can find a solution by pushing the orange icon in the top left of this notebook.
