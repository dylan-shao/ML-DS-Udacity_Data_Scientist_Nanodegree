
### PCA Mini Project

In the lesson, you saw how you could use PCA to substantially reduce the dimensionality of the handwritten digits.  In this mini-project, you will be using the **cars.csv** file.  

To begin, run the cell below to read in the necessary libraries and the dataset.  I also read in the helper functions that you used throughout the lesson in case you might find them helpful in completing this project.  Otherwise, you can always create functions of your own!


```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from helper_functions import do_pca, scree_plot, plot_components, pca_results
from IPython import display
import test_code2 as t

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

df = pd.read_csv('./data/cars.csv')
```

`1.` Now your data is stored in **df**.  Use the below cells to take a look your dataset.  At the end of your exploration, use your findings to match the appropriate variable to each key in the dictionary below.  


```python
#Use this cell for work
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 387 entries, Acura 3.5 RL to Volvo XC90 T6
    Data columns (total 18 columns):
    Sports        387 non-null int64
    SUV           387 non-null int64
    Wagon         387 non-null int64
    Minivan       387 non-null int64
    Pickup        387 non-null int64
    AWD           387 non-null int64
    RWD           387 non-null int64
    Retail        387 non-null int64
    Dealer        387 non-null int64
    Engine        387 non-null float64
    Cylinders     387 non-null int64
    Horsepower    387 non-null int64
    CityMPG       387 non-null int64
    HighwayMPG    387 non-null int64
    Weight        387 non-null int64
    Wheelbase     387 non-null int64
    Length        387 non-null int64
    Width         387 non-null int64
    dtypes: float64(1), int64(17)
    memory usage: 57.4+ KB



```python
# and this one
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
      <th>Sports</th>
      <th>SUV</th>
      <th>Wagon</th>
      <th>Minivan</th>
      <th>Pickup</th>
      <th>AWD</th>
      <th>RWD</th>
      <th>Retail</th>
      <th>Dealer</th>
      <th>Engine</th>
      <th>Cylinders</th>
      <th>Horsepower</th>
      <th>CityMPG</th>
      <th>HighwayMPG</th>
      <th>Weight</th>
      <th>Wheelbase</th>
      <th>Length</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.0</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
      <td>387.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.116279</td>
      <td>0.152455</td>
      <td>0.072351</td>
      <td>0.054264</td>
      <td>0.0</td>
      <td>0.201550</td>
      <td>0.242894</td>
      <td>33231.180879</td>
      <td>30440.653747</td>
      <td>3.127390</td>
      <td>5.757106</td>
      <td>214.444444</td>
      <td>20.312661</td>
      <td>27.263566</td>
      <td>3532.457364</td>
      <td>107.211886</td>
      <td>184.961240</td>
      <td>71.276486</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.320974</td>
      <td>0.359926</td>
      <td>0.259404</td>
      <td>0.226830</td>
      <td>0.0</td>
      <td>0.401677</td>
      <td>0.429387</td>
      <td>19724.634576</td>
      <td>17901.179282</td>
      <td>1.014314</td>
      <td>1.490182</td>
      <td>70.262822</td>
      <td>5.262333</td>
      <td>5.636005</td>
      <td>706.003622</td>
      <td>7.086553</td>
      <td>13.237999</td>
      <td>3.368329</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10280.000000</td>
      <td>9875.000000</td>
      <td>1.400000</td>
      <td>3.000000</td>
      <td>73.000000</td>
      <td>10.000000</td>
      <td>12.000000</td>
      <td>1850.000000</td>
      <td>89.000000</td>
      <td>143.000000</td>
      <td>64.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20997.000000</td>
      <td>19575.000000</td>
      <td>2.300000</td>
      <td>4.000000</td>
      <td>165.000000</td>
      <td>18.000000</td>
      <td>24.000000</td>
      <td>3107.000000</td>
      <td>103.000000</td>
      <td>177.000000</td>
      <td>69.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>28495.000000</td>
      <td>26155.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>210.000000</td>
      <td>19.000000</td>
      <td>27.000000</td>
      <td>3469.000000</td>
      <td>107.000000</td>
      <td>186.000000</td>
      <td>71.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>39552.500000</td>
      <td>36124.000000</td>
      <td>3.800000</td>
      <td>6.000000</td>
      <td>250.000000</td>
      <td>21.500000</td>
      <td>30.000000</td>
      <td>3922.000000</td>
      <td>112.000000</td>
      <td>193.000000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>192465.000000</td>
      <td>173560.000000</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>493.000000</td>
      <td>60.000000</td>
      <td>66.000000</td>
      <td>6400.000000</td>
      <td>130.000000</td>
      <td>221.000000</td>
      <td>81.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# and this one if you need it - and create more cells if you need them
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
      <th>Sports</th>
      <th>SUV</th>
      <th>Wagon</th>
      <th>Minivan</th>
      <th>Pickup</th>
      <th>AWD</th>
      <th>RWD</th>
      <th>Retail</th>
      <th>Dealer</th>
      <th>Engine</th>
      <th>Cylinders</th>
      <th>Horsepower</th>
      <th>CityMPG</th>
      <th>HighwayMPG</th>
      <th>Weight</th>
      <th>Wheelbase</th>
      <th>Length</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Acura 3.5 RL</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43755</td>
      <td>39014</td>
      <td>3.5</td>
      <td>6</td>
      <td>225</td>
      <td>18</td>
      <td>24</td>
      <td>3880</td>
      <td>115</td>
      <td>197</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Acura 3.5 RL Navigation</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46100</td>
      <td>41100</td>
      <td>3.5</td>
      <td>6</td>
      <td>225</td>
      <td>18</td>
      <td>24</td>
      <td>3893</td>
      <td>115</td>
      <td>197</td>
      <td>72</td>
    </tr>
    <tr>
      <th>Acura MDX</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>36945</td>
      <td>33337</td>
      <td>3.5</td>
      <td>6</td>
      <td>265</td>
      <td>17</td>
      <td>23</td>
      <td>4451</td>
      <td>106</td>
      <td>189</td>
      <td>77</td>
    </tr>
    <tr>
      <th>Acura NSX S</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>89765</td>
      <td>79978</td>
      <td>3.2</td>
      <td>6</td>
      <td>290</td>
      <td>17</td>
      <td>24</td>
      <td>3153</td>
      <td>100</td>
      <td>174</td>
      <td>71</td>
    </tr>
    <tr>
      <th>Acura RSX</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23820</td>
      <td>21761</td>
      <td>2.0</td>
      <td>4</td>
      <td>200</td>
      <td>24</td>
      <td>31</td>
      <td>2778</td>
      <td>101</td>
      <td>172</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.shape)
print(np.sum(df['Minivan']) / df.shape[0])
print(np.max(df['HighwayMPG']))
```

    (387, 18)
    0.05426356589147287
    66



```python
a = 7
b = 66
c = 387
d = 18
e = 0.23
f = 0.05

solution_1_dict = {
    'The number of cars in the dataset': c, #letter here,
    'The number of car features in the dataset': d, #letter here,
    'The number of dummy variables in the dataset': a, #letter here,
    'The proportion of minivans in the dataset': f, #letter here,
    'The max highway mpg for any car': b#letter here
}
```


```python
# Check your solution against ours by running this cell
display.HTML(t.check_question_one(solution_1_dict))
```

    Nice job!  Looks like your dataset matches what we found!





<img src="https://bit.ly/2K9X0gD">



`2.` There are some particularly nice properties about PCA to keep in mind.  Use the dictionary below to match the correct variable as the key to each statement.  When you are ready, check your solution against ours by running the following cell.


```python
a = True
b = False

solution_2_dict = {
    'The components span the directions of maximum variability.':a, #letter here,
    'The components are always orthogonal to one another.': a,#letter here,
    'Eigenvalues tell us the amount of information a component holds': a,#letter here
}
```


```python
# Check your solution against ours by running this cell
t.check_question_two(solution_2_dict)
```

    That's right these are all true.  Principal components are orthogonal, span the directions of maximum variability, and the corresponding eigenvalues tell us how much of the original variability is explained by each component.


`3.` Fit PCA to reduce the current dimensionality of the datset to 3 dimensions.  You can use the helper functions, or perform the steps on your own.  If you fit on your own, be sure to standardize your data.  At the end of this process, you will want an **X**  matrix with the reduced dimensionality to only 3 features.  Additionally, you will want your **pca** object back that has been used to fit and transform your dataset. 


```python
#Scale your data, fit, and transform using pca
#you need a pca object and your transformed data matrix

# pca, df_pca = do_pca(3, df)

df_ss = StandardScaler().fit_transform(df)
pca = PCA(3)
df_pca = pca.fit_transform(df_ss)
```

`4.` Once you have your pca object, you can take a closer look at what comprises each of the principal components.  Use the **pca_results** function from the **helper_functions** module assist with taking a closer look at the results of your analysis.  The function takes two arguments: the full dataset and the pca object you created.


```python
pca_results(df, pca)
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
      <th>Explained Variance</th>
      <th>Sports</th>
      <th>SUV</th>
      <th>Wagon</th>
      <th>Minivan</th>
      <th>Pickup</th>
      <th>AWD</th>
      <th>RWD</th>
      <th>Retail</th>
      <th>Dealer</th>
      <th>Engine</th>
      <th>Cylinders</th>
      <th>Horsepower</th>
      <th>CityMPG</th>
      <th>HighwayMPG</th>
      <th>Weight</th>
      <th>Wheelbase</th>
      <th>Length</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dimension 1</th>
      <td>0.4352</td>
      <td>-0.0343</td>
      <td>-0.1298</td>
      <td>0.0289</td>
      <td>-0.0481</td>
      <td>-0.0</td>
      <td>-0.0928</td>
      <td>-0.1175</td>
      <td>-0.2592</td>
      <td>-0.2576</td>
      <td>-0.3396</td>
      <td>-0.3263</td>
      <td>-0.3118</td>
      <td>0.3063</td>
      <td>0.3061</td>
      <td>-0.3317</td>
      <td>-0.2546</td>
      <td>-0.2414</td>
      <td>-0.2886</td>
    </tr>
    <tr>
      <th>Dimension 2</th>
      <td>0.1667</td>
      <td>0.4420</td>
      <td>-0.2261</td>
      <td>-0.0106</td>
      <td>-0.2074</td>
      <td>0.0</td>
      <td>-0.1447</td>
      <td>0.3751</td>
      <td>0.3447</td>
      <td>0.3453</td>
      <td>0.0022</td>
      <td>0.0799</td>
      <td>0.2342</td>
      <td>0.0169</td>
      <td>0.0433</td>
      <td>-0.1832</td>
      <td>-0.3066</td>
      <td>-0.2701</td>
      <td>-0.2163</td>
    </tr>
    <tr>
      <th>Dimension 3</th>
      <td>0.1034</td>
      <td>0.0875</td>
      <td>0.4898</td>
      <td>0.0496</td>
      <td>-0.2818</td>
      <td>-0.0</td>
      <td>0.5506</td>
      <td>-0.2416</td>
      <td>0.0154</td>
      <td>0.0132</td>
      <td>-0.0489</td>
      <td>-0.0648</td>
      <td>0.0040</td>
      <td>-0.1421</td>
      <td>-0.2486</td>
      <td>0.0851</td>
      <td>-0.2846</td>
      <td>-0.3361</td>
      <td>-0.1369</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_15_1.png)


`5.` Use the results, to match each of the variables as the value to the most appropriate key in the dictionary below.  When you are ready to check your answers, run the following cell to see if your solution matches ours!


```python
a = 'car weight'
b = 'sports cars'
c = 'gas mileage'
d = 0.4352
e = 0.3061
f = 0.1667
g = 0.7053

solution_5_dict = {
    'The first component positively weights items related to': c, #letter here, 
    'The amount of variability explained by the first component is': d,#letter here,
    'The largest weight of the second component is related to': b,#letter here,
    'The total amount of variability explained by the first three components': g#letter here
}
```


```python
# Run this cell to check if your solution matches ours.
t.check_question_five(solution_5_dict)
```

    That's right!  Looks like you know a lot about PCA!


`6.` How many components need to be kept to explain at least 85% of the variability in the original dataset?  When you think you have the answer, store it in the variable `num_comps`.  Then run the following cell to see if your solution matches ours!


```python
#Code to find number of components providing more than 
# 85% of variance explained

for i in range(3,15):
    pca, X_pca = do_pca(i, df)
    print(i, np.sum(pca.explained_variance_ratio_))
    
num_comps = 6#num components stored here
```

    3 0.7054128660268958
    4 0.7685122291942386
    5 0.8240656153204096
    6 0.8682760591473635
    7 0.8971575538109497
    8 0.9249693047545744
    9 0.9498825459544127
    10 0.9680096689137985
    11 0.9770081664016601
    12 0.9845815494404042
    13 0.9903628982892551
    14 0.9945773134492496



```python
# Now check your answer here to complete this mini project!
display.HTML(t.question_check_six(num_comps))
```

    Nice job!  That's right!  With 6 components, you can explain more than 85% of the variability in the original dataset.





<img src="https://bit.ly/2cKTiso">


