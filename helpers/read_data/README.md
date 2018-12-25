

```python
import pandas as pd
import numpy as np
from pandas import read_csv


train_data = read_csv('../data/dataNoLabel.csv',header=None)
X = train_data.iloc[:,0:6]
y1 = train_data.iloc[:,-1:]
y2 = train_data.iloc[:,-1]

print(train_data.shape)
print('.iloc[:,0:6]', X.shape, type(X))
print('.iloc[:,-1:]', y1.shape, type(y1))
print('.iloc[:,-1]',y2.shape, type(y2))
```

    (100, 7)
    .iloc[:,0:6] (100, 6) <class 'pandas.core.frame.DataFrame'>
    .iloc[:,-1:] (100, 1) <class 'pandas.core.frame.DataFrame'>
    .iloc[:,-1] (100,) <class 'pandas.core.series.Series'>



```python
# get the index of a pandas series from the value
# 
# series = dataframe['column_name']
# series[series == 'series_value'].index[0]
```


```python
a = pd.Series([1,2,3,4,5,0,4,3,0])
a = a.replace([0], np.nan)

a.isnull().sum()
```




    2


