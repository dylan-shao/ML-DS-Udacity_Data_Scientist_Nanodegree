
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

plt.rcParams['figure.figsize'] = (16, 9)
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
data
```




    array([[ 650.56533474,   49.3417269 ],
           [ 512.89427271,   70.23141424],
           [ 885.05745295,   49.21326946],
           [1028.64120981,   63.47194594],
           [ 746.89919481,   55.38739564],
           [ 613.2333586 ,   44.57559963],
           [ 444.75178714,   63.98918574],
           [ 930.93660623,   46.43849003],
           [ 437.5222003 ,   51.57850909],
           [ 606.67650723,   28.19454629],
           [ 780.65576627,   50.75599161],
           [ 486.73387708,   70.6114148 ],
           [ 572.29300794,   52.32021702],
           [ 736.4990739 ,   47.37200171],
           [ 890.47697778,   47.88985671],
           [ 770.86222021,   45.56039875],
           [ 258.12077446,   64.00104648],
           [ 966.05481522,   61.91109395],
           [ 619.33670008,   34.40282015],
           [ 529.3610375 ,   35.92841471],
           [ 475.74709941,   37.45249458],
           [ 735.45725023,   55.59920618],
           [ 851.56091963,   50.00454753],
           [ 244.16694184,   48.61286636],
           [ 815.63747853,   73.08911878],
           [ 318.61196234,   67.2902949 ],
           [ 366.01912005,   64.6317272 ],
           [ 703.14122862,   52.70215089],
           [ 230.0334028 ,   60.62458903],
           [ 340.29281254,   56.42758993],
           [ 839.89971571,   26.50148998],
           [ 456.49686245,   63.85829726],
           [1055.14622857,   69.04277165],
           [ 966.38249534,   82.88344771],
           [ 214.78020175,   54.57965216],
           [ 473.95302013,   53.93805556],
           [ 842.53963024,   73.3140027 ],
           [ 522.673593  ,   32.67454885],
           [ 859.33611855,   52.76253997],
           [ 241.74368465,   65.51745634],
           [ 807.99235982,   26.22993968],
           [ 331.76912259,   62.8777666 ],
           [ 786.11358422,   42.16987558],
           [ 524.72158872,   46.00325147],
           [ 965.65484399,   70.11648245],
           [ 526.5546184 ,   54.54302459],
           [ 375.03644066,   55.52487885],
           [ 680.98251062,   28.1455318 ],
           [ 326.22094683,   49.31475853],
           [ 159.37955054,   71.85537885],
           [ 664.85248234,   72.71264551],
           [ 305.35803476,   53.88693934],
           [ 765.88876359,   71.24385008],
           [ 543.09107833,   47.90734878],
           [ 398.77196134,   81.36441523],
           [ 311.09042353,   49.25249421],
           [ 474.65938165,   45.50198326],
           [ 713.21953425,   39.71824092],
           [ 321.12507169,   54.83633095],
           [ 344.8575339 ,   44.53115566],
           [ 788.79039184,   60.83956521],
           [ 372.0454602 ,   43.18917434],
           [ 598.00273157,   53.6428767 ],
           [ 885.73439645,   48.8163953 ],
           [ 796.60071199,   48.35224059],
           [ 668.05674956,   50.44340658],
           [ 357.75751225,   54.47374841],
           [ 168.66527109,   65.76741254],
           [ 786.32080232,   53.65338743],
           [ 177.07310431,   66.33535511],
           [ 218.51136537,   45.14779235],
           [ 861.42352748,   60.91052955],
           [ 341.72216986,   48.10637129],
           [ 754.14136559,   25.14562639],
           [ 604.26731472,   42.55438702],
           [ 636.45919239,   39.30194631],
           [ 356.89863382,   66.6948656 ],
           [ 297.8672009 ,   57.88709814],
           [ 636.45795048,   41.75041116],
           [ 298.83718609,   63.40843829],
           [ 252.26948478,   57.6893334 ],
           [ 485.39725804,   47.82160469],
           [1012.08945312,   53.02349685],
           [ 851.41614041,   60.64613373],
           [ 874.13317916,   54.18753059],
           [ 829.12303755,   51.19953697],
           [ 883.30840569,   74.71732138],
           [ 242.21505548,   53.30065632],
           [ 469.80875547,   64.96658715],
           [ 905.60089599,   63.81261406],
           [ 683.40553512,   41.60687172],
           [ 392.45683651,   52.97607599],
           [ 308.92154052,   39.60129625],
           [ 460.51670665,   54.25494406],
           [ 465.28807281,   39.68669239],
           [ 955.67719779,   59.52475725],
           [ 334.84151463,   73.72809824],
           [ 489.03716866,   41.542038  ],
           [ 643.5807763 ,   40.35363946],
           [ 290.13057762,   45.09090808],
           [ 753.28489495,   70.48153824],
           [ 928.83932281,   57.15725344],
           [ 285.19121396,   48.90805636],
           [ 708.77494415,   31.95899912],
           [ 221.3077346 ,   68.40550768],
           [ 441.64160509,   34.19923594],
           [ 831.89812424,   63.03303311],
           [ 719.48964358,   26.35358162],
           [ 571.6463439 ,   40.88876146],
           [1043.27554154,   72.24262806],
           [ 489.97616305,   64.45689317],
           [ 637.46399122,   43.27150636],
           [ 216.55793335,   81.43280078],
           [ 814.81837649,   52.66912401],
           [ 303.89578268,   64.19528916],
           [1096.22234809,   52.21631156],
           [ 738.71841488,   31.38929689],
           [ 423.41154556,   59.6089886 ],
           [ 296.9836394 ,   72.65697792],
           [ 262.48459054,   57.99591907],
           [ 668.51452991,   64.73339073],
           [ 511.41729509,   35.29980066],
           [ 364.93425111,   32.60862759],
           [ 876.25231984,   83.25807536],
           [ 849.65717809,   76.86474574],
           [ 419.33638842,   72.62018613],
           [ 217.79341916,   44.08540631],
           [ 545.33960536,   59.36592063],
           [ 375.60214   ,   45.60837982],
           [ 147.81981006,   61.58335359],
           [ 592.56254277,   48.85325789],
           [ 690.05287851,   35.76508967],
           [ 104.35488541,   56.29835927],
           [ 881.38175824,   77.86806637],
           [ 967.80495988,   67.23507931],
           [ 560.32549655,   31.1362388 ],
           [ 681.53471721,   45.27498287],
           [ 473.56831018,   55.97539301],
           [ 241.21200737,   57.86990683],
           [ 561.2716042 ,   41.21365689],
           [ 272.11076209,   49.47680088],
           [ 352.20287434,   58.96167278],
           [1015.85609637,   53.15958546],
           [ 303.23956017,   61.28820997],
           [ 456.27771268,   47.28109428],
           [ 383.7383672 ,   58.67345025],
           [ 395.98541146,   68.09094231],
           [ 798.06681334,   63.57966427],
           [ 676.61098459,   42.38747196],
           [ 731.96716778,   51.86454135],
           [ 361.55325971,   44.87984553],
           [ 885.68408926,   61.33297729],
           [ 546.19389962,   55.65444232],
           [ 895.28676019,   69.31303755],
           [ 765.24524057,   43.0698165 ],
           [ 768.02168831,   60.81511278],
           [ 984.43451381,   68.44539681],
           [ 489.7429227 ,   61.49272644],
           [ 511.31164608,   60.68413155],
           [ 579.84744152,   61.89371985],
           [ 169.66718007,   67.76885468],
           [ 439.44907202,   48.44073427],
           [ 499.76661966,   30.67048915],
           [ 894.70879051,   44.99092104],
           [ 272.88695109,   44.00807886],
           [ 630.87366805,   39.30056162],
           [ 493.29513095,   36.87718027],
           [ 385.66255439,   57.44766295],
           [ 659.86353234,   43.89170821],
           [ 491.65696356,   58.82735356],
           [ 756.06628608,   67.6692234 ],
           [ 250.40841664,   56.95959663],
           [ 759.49309004,   57.01798053],
           [ 652.26117052,   42.40432327],
           [ 578.27016507,   62.57883651],
           [ 543.83539021,   24.60163867],
           [ 470.01090462,   35.00029088],
           [ 310.9836313 ,   45.15733851],
           [ 267.04833347,   40.29536019],
           [ 814.05111455,   36.62829561],
           [ 297.26153159,   69.96612344],
           [ 767.21479296,   60.5395393 ],
           [ 874.06516581,   48.29798031],
           [ 821.42079396,   63.15553873],
           [ 801.29114076,   60.59443866],
           [ 103.13035783,   55.99433691],
           [ 516.18204018,   58.05034455],
           [ 481.30597621,   68.53889043],
           [ 379.30851182,   48.48301582],
           [ 254.50236622,   76.94221922],
           [ 214.7561598 ,   63.0394181 ],
           [ 760.99938229,   49.56214968],
           [ 973.6742428 ,   51.33315635],
           [ 959.74604182,   57.02162153],
           [ 226.72353515,   66.94489704],
           [ 852.7420191 ,   53.10058099],
           [ 652.60647374,   44.98499246],
           [  92.99848075,   79.38689731],
           [ 735.15590569,   47.52689898],
           [ 366.5197166 ,   55.30870618]])



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


![png](output_6_0.png)


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


![png](output_9_0.png)


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


![png](output_11_0.png)



```python
#plot each of the scaled datasets
labels = fit_kmeans(df_ss, 10) #fit kmeans to get the labels
plt.scatter(df_ss['height'], df_ss['weight'], c=labels, cmap='Set1');
```


![png](output_12_0.png)



```python
#another plot of the other scaled dataset
labels = fit_kmeans(df_min_max, 10)
plt.scatter(df_min_max['height'], df_min_max['weight'], c=labels, cmap='Set1');
```


![png](output_13_0.png)


Write your response here!

Different scaling will give you different clusters! 
