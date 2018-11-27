
### Feature Scaling

With any distance based machine learning model (regularized regression methods, neural networks, and now kmeans), you will want to scale your data.  

If you have some features that are on completely different scales, this can greatly impact the clusters you get when using K-Means. 

In this notebook, you will get to see this first hand.  To begin, let's read in the necessary libraries.


```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing as p

%matplotlib inline

plt.rcParams['figure.figsize'] = (8, 6)
import helpers2 as h
import tests as t


# Create the dataset for the notebook
data = h.simulate_data(200, 2, 4)
df = pd.DataFrame(data)
df.columns = ['height', 'weight']
df['height'] = np.abs(df['height']*100)
df['weight'] = df['weight'] + np.random.normal(50, 10, 200)
```

`1.` Next, take a look at the data to get familiar with it.  The dataset has two columns, and it is stored in the **df** variable.  It might be useful to get an idea of the spread in the current data, as well as a visual of the points.  


```python
#Take a look at the data
df.describe()
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
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>569.726207</td>
      <td>52.268325</td>
    </tr>
    <tr>
      <th>std</th>
      <td>246.966215</td>
      <td>12.660423</td>
    </tr>
    <tr>
      <th>min</th>
      <td>92.998481</td>
      <td>15.021845</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>357.542793</td>
      <td>44.460751</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>545.766752</td>
      <td>52.582614</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>773.310607</td>
      <td>60.373623</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1096.222348</td>
      <td>86.778607</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
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
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>650.565335</td>
      <td>62.036553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>512.894273</td>
      <td>45.853920</td>
    </tr>
    <tr>
      <th>2</th>
      <td>885.057453</td>
      <td>55.460256</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1028.641210</td>
      <td>76.908892</td>
    </tr>
    <tr>
      <th>4</th>
      <td>746.899195</td>
      <td>37.888562</td>
    </tr>
  </tbody>
</table>
</div>



Now that we've got a dataset, let's look at some options for scaling the data.  As well as how the data might be scaled.  There are two very common types of feature scaling that we should discuss:


**I.  MinMaxScaler**

In some cases it is useful to think of your data in terms of the percent they are as compared to the maximum value.  In these cases, you will want to use **MinMaxScaler**.

**II. StandardScaler**

Another very popular type of scaling is to scale data so that it has mean 0 and variance 1.  In these cases, you will want to use **StandardScaler**.  

It is probably more appropriate with this data to use **StandardScaler**.  However, to get practice with feature scaling methods in python, we will perform both.

`2.` First let's fit the **StandardScaler** transformation to this dataset.  I will do this one so you can see how to apply preprocessing in sklearn.


```python
df_ss = p.StandardScaler().fit_transform(df) # Fit and transform the data
```


```python
df_ss = pd.DataFrame(df_ss) #create a dataframe
df_ss.columns = ['height', 'weight'] #add column names again

plt.scatter(df_ss['height'], df_ss['weight']); # create a plot
```


![png](output_7_0.png)


`3.` Now it's your turn.  Try fitting the **MinMaxScaler** transformation to this dataset. You should be able to use the previous example to assist.


```python
# fit and transform
df_min_max = p.MinMaxScaler().fit_transform(df)
```


```python
#create a dataframe
#change the column names
#plot the data
df_min_max = pd.DataFrame(df_min_max) #create a dataframe
df_min_max.columns = ['height', 'weight'] #add column names again

plt.scatter(df_min_max['height'], df_min_max['weight']); # create a plot
```


![png](output_10_0.png)


`4.`  Now let's take a look at how kmeans divides the dataset into different groups for each of the different scalings of the data.  Did you end up with different clusters when the data was scaled differently?


```python
def fit_kmeans(data, centers):
    '''
    INPUT:
        data = the dataset you would like to fit kmeans to (dataframe)
        centers = the number of centroids (int)
    OUTPUT:
        labels - the labels for each datapoint to which group it belongs (nparray)
    
    '''
    kmeans = KMeans(centers)
    labels = kmeans.fit_predict(data)
    return labels

labels = fit_kmeans(df, 10) #fit kmeans to get the labels
    
# Plot the original data with clusters
plt.scatter(df['height'], df['weight'], c=labels, cmap='Set1');
```


![png](output_12_0.png)



```python
#plot each of the scaled datasets
labels = fit_kmeans(df_ss, 10) #fit kmeans to get the labels
plt.scatter(df_ss['height'], df_ss['weight'], c=labels, cmap='Set1');
```


![png](output_13_0.png)



```python
#another plot of the other scaled dataset
labels = fit_kmeans(df_min_max, 10)
plt.scatter(df_min_max['height'], df_min_max['weight'], c=labels, cmap='Set1');
```


![png](output_14_0.png)


Write your response here!

Different scaling will give you different clusters! But the minmax and standard one looks similiar.
