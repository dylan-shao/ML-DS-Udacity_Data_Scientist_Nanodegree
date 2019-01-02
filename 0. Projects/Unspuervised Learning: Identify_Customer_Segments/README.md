
# Project: Identify Customer Segments

In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.

This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.

It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.

At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**


```python
# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn import preprocessing as p
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# magic word for producing visualizations in notebook
%matplotlib inline
```

### Step 0: Load the Data

There are four files associated with this project (not including this one):

- `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
- `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.

To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.

Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.


```python
# those data is ignored on git as this is confiditial

# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', delimiter=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', delimiter=';')
```


```python
# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).

# azdias.describe()
```


```python
azdias.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891221 entries, 0 to 891220
    Data columns (total 85 columns):
    AGER_TYP                 891221 non-null int64
    ALTERSKATEGORIE_GROB     891221 non-null int64
    ANREDE_KZ                891221 non-null int64
    CJT_GESAMTTYP            886367 non-null float64
    FINANZ_MINIMALIST        891221 non-null int64
    FINANZ_SPARER            891221 non-null int64
    FINANZ_VORSORGER         891221 non-null int64
    FINANZ_ANLEGER           891221 non-null int64
    FINANZ_UNAUFFAELLIGER    891221 non-null int64
    FINANZ_HAUSBAUER         891221 non-null int64
    FINANZTYP                891221 non-null int64
    GEBURTSJAHR              891221 non-null int64
    GFK_URLAUBERTYP          886367 non-null float64
    GREEN_AVANTGARDE         891221 non-null int64
    HEALTH_TYP               891221 non-null int64
    LP_LEBENSPHASE_FEIN      886367 non-null float64
    LP_LEBENSPHASE_GROB      886367 non-null float64
    LP_FAMILIE_FEIN          886367 non-null float64
    LP_FAMILIE_GROB          886367 non-null float64
    LP_STATUS_FEIN           886367 non-null float64
    LP_STATUS_GROB           886367 non-null float64
    NATIONALITAET_KZ         891221 non-null int64
    PRAEGENDE_JUGENDJAHRE    891221 non-null int64
    RETOURTYP_BK_S           886367 non-null float64
    SEMIO_SOZ                891221 non-null int64
    SEMIO_FAM                891221 non-null int64
    SEMIO_REL                891221 non-null int64
    SEMIO_MAT                891221 non-null int64
    SEMIO_VERT               891221 non-null int64
    SEMIO_LUST               891221 non-null int64
    SEMIO_ERL                891221 non-null int64
    SEMIO_KULT               891221 non-null int64
    SEMIO_RAT                891221 non-null int64
    SEMIO_KRIT               891221 non-null int64
    SEMIO_DOM                891221 non-null int64
    SEMIO_KAEM               891221 non-null int64
    SEMIO_PFLICHT            891221 non-null int64
    SEMIO_TRADV              891221 non-null int64
    SHOPPER_TYP              891221 non-null int64
    SOHO_KZ                  817722 non-null float64
    TITEL_KZ                 817722 non-null float64
    VERS_TYP                 891221 non-null int64
    ZABEOTYP                 891221 non-null int64
    ALTER_HH                 817722 non-null float64
    ANZ_PERSONEN             817722 non-null float64
    ANZ_TITEL                817722 non-null float64
    HH_EINKOMMEN_SCORE       872873 non-null float64
    KK_KUNDENTYP             306609 non-null float64
    W_KEIT_KIND_HH           783619 non-null float64
    WOHNDAUER_2008           817722 non-null float64
    ANZ_HAUSHALTE_AKTIV      798073 non-null float64
    ANZ_HH_TITEL             794213 non-null float64
    GEBAEUDETYP              798073 non-null float64
    KONSUMNAEHE              817252 non-null float64
    MIN_GEBAEUDEJAHR         798073 non-null float64
    OST_WEST_KZ              798073 non-null object
    WOHNLAGE                 798073 non-null float64
    CAMEO_DEUG_2015          792242 non-null object
    CAMEO_DEU_2015           792242 non-null object
    CAMEO_INTL_2015          792242 non-null object
    KBA05_ANTG1              757897 non-null float64
    KBA05_ANTG2              757897 non-null float64
    KBA05_ANTG3              757897 non-null float64
    KBA05_ANTG4              757897 non-null float64
    KBA05_BAUMAX             757897 non-null float64
    KBA05_GBZ                757897 non-null float64
    BALLRAUM                 797481 non-null float64
    EWDICHTE                 797481 non-null float64
    INNENSTADT               797481 non-null float64
    GEBAEUDETYP_RASTER       798066 non-null float64
    KKK                      770025 non-null float64
    MOBI_REGIO               757897 non-null float64
    ONLINE_AFFINITAET        886367 non-null float64
    REGIOTYP                 770025 non-null float64
    KBA13_ANZAHL_PKW         785421 non-null float64
    PLZ8_ANTG1               774706 non-null float64
    PLZ8_ANTG2               774706 non-null float64
    PLZ8_ANTG3               774706 non-null float64
    PLZ8_ANTG4               774706 non-null float64
    PLZ8_BAUMAX              774706 non-null float64
    PLZ8_HHZ                 774706 non-null float64
    PLZ8_GBZ                 774706 non-null float64
    ARBEIT                   794005 non-null float64
    ORTSGR_KLS9              794005 non-null float64
    RELAT_AB                 794005 non-null float64
    dtypes: float64(49), int64(32), object(4)
    memory usage: 578.0+ MB



```python
azdias.head()
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
      <th>AGER_TYP</th>
      <th>ALTERSKATEGORIE_GROB</th>
      <th>ANREDE_KZ</th>
      <th>CJT_GESAMTTYP</th>
      <th>FINANZ_MINIMALIST</th>
      <th>FINANZ_SPARER</th>
      <th>FINANZ_VORSORGER</th>
      <th>FINANZ_ANLEGER</th>
      <th>FINANZ_UNAUFFAELLIGER</th>
      <th>FINANZ_HAUSBAUER</th>
      <th>...</th>
      <th>PLZ8_ANTG1</th>
      <th>PLZ8_ANTG2</th>
      <th>PLZ8_ANTG3</th>
      <th>PLZ8_ANTG4</th>
      <th>PLZ8_BAUMAX</th>
      <th>PLZ8_HHZ</th>
      <th>PLZ8_GBZ</th>
      <th>ARBEIT</th>
      <th>ORTSGR_KLS9</th>
      <th>RELAT_AB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>2</td>
      <td>1</td>
      <td>2.0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1</td>
      <td>1</td>
      <td>2</td>
      <td>5.0</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1</td>
      <td>3</td>
      <td>2</td>
      <td>3.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2.0</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1</td>
      <td>3</td>
      <td>1</td>
      <td>5.0</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>




```python
feat_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 85 entries, 0 to 84
    Data columns (total 4 columns):
    attribute             85 non-null object
    information_level     85 non-null object
    type                  85 non-null object
    missing_or_unknown    85 non-null object
    dtypes: object(4)
    memory usage: 2.7+ KB



```python
feat_info.head(8)
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
      <th>attribute</th>
      <th>information_level</th>
      <th>type</th>
      <th>missing_or_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AGER_TYP</td>
      <td>person</td>
      <td>categorical</td>
      <td>[-1,0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ALTERSKATEGORIE_GROB</td>
      <td>person</td>
      <td>ordinal</td>
      <td>[-1,0,9]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ANREDE_KZ</td>
      <td>person</td>
      <td>categorical</td>
      <td>[-1,0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CJT_GESAMTTYP</td>
      <td>person</td>
      <td>categorical</td>
      <td>[0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FINANZ_MINIMALIST</td>
      <td>person</td>
      <td>ordinal</td>
      <td>[-1]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FINANZ_SPARER</td>
      <td>person</td>
      <td>ordinal</td>
      <td>[-1]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FINANZ_VORSORGER</td>
      <td>person</td>
      <td>ordinal</td>
      <td>[-1]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>FINANZ_ANLEGER</td>
      <td>person</td>
      <td>ordinal</td>
      <td>[-1]</td>
    </tr>
  </tbody>
</table>
</div>



> **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 

## Step 1: Preprocessing

### Step 1.1: Assess Missing Data

The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!

#### Step 1.1.1: Convert Missing Value Codes to NaNs
The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.

**As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**


```python
# test
# series = feat_info['attribute']
# # print(series[series == 'HEALTH_TYP'].index[0])

# # series[15]
# missing_or_unknow_list = list(map(lambda x: int(x), feat_info['missing_or_unknown'][0][1:-1].split(',')))
# missing_or_unknow_count = 0

# azdias['AGER_TYP'] = azdias['AGER_TYP'].replace(missing_or_unknow_list, np.nan)
# azdias['AGER_TYP'].isnull().sum() / len(azdias['AGER_TYP'])
# for index, value in azdias['LP_LEBENSPHASE_FEIN'].iteritems():
#     if value in list(map(lambda x: int(x), missing_or_unknow_list)):
#             azdias_column[c_index] = np.nan
#             missing_or_unknow_count += 1
```


```python
type(np.NaN)
```




    float




```python
# Identify missing or unknown data values and convert them to NaNs.

for index, row in feat_info.iterrows():
    column_name = row['attribute']
    missing_or_unknow_list = row['missing_or_unknown'][1:-1].split(',')
    print(column_name)
    
    missing_or_unknow_count = 0
    if azdias[column_name].dtypes != 'object' and len(missing_or_unknow_list) != 0 and missing_or_unknow_list[0] != '':
        missing_or_unknow_list = list(map(float, missing_or_unknow_list))
    print(missing_or_unknow_list)
    azdias[column_name] = azdias[column_name].replace(missing_or_unknow_list, np.nan)
    percentage = azdias[column_name].isnull().sum() / len(azdias[column_name])
    
    print('The percentage of missing value for {} is {}'.format(column_name, percentage))

azdias_filled = azdias

```

    AGER_TYP
    [-1.0, 0.0]
    The percentage of missing value for AGER_TYP is 0.7695543529607134
    ALTERSKATEGORIE_GROB
    [-1.0, 0.0, 9.0]
    The percentage of missing value for ALTERSKATEGORIE_GROB is 0.0032326437550282143
    ANREDE_KZ
    [-1.0, 0.0]
    The percentage of missing value for ANREDE_KZ is 0.0
    CJT_GESAMTTYP
    [0.0]
    The percentage of missing value for CJT_GESAMTTYP is 0.005446460529992
    FINANZ_MINIMALIST
    [-1.0]
    The percentage of missing value for FINANZ_MINIMALIST is 0.0
    FINANZ_SPARER
    [-1.0]
    The percentage of missing value for FINANZ_SPARER is 0.0
    FINANZ_VORSORGER
    [-1.0]
    The percentage of missing value for FINANZ_VORSORGER is 0.0
    FINANZ_ANLEGER
    [-1.0]
    The percentage of missing value for FINANZ_ANLEGER is 0.0
    FINANZ_UNAUFFAELLIGER
    [-1.0]
    The percentage of missing value for FINANZ_UNAUFFAELLIGER is 0.0
    FINANZ_HAUSBAUER
    [-1.0]
    The percentage of missing value for FINANZ_HAUSBAUER is 0.0
    FINANZTYP
    [-1.0]
    The percentage of missing value for FINANZTYP is 0.0
    GEBURTSJAHR
    [0.0]
    The percentage of missing value for GEBURTSJAHR is 0.4402028228688507
    GFK_URLAUBERTYP
    ['']
    The percentage of missing value for GFK_URLAUBERTYP is 0.005446460529992
    GREEN_AVANTGARDE
    ['']
    The percentage of missing value for GREEN_AVANTGARDE is 0.0
    HEALTH_TYP
    [-1.0, 0.0]
    The percentage of missing value for HEALTH_TYP is 0.12476815514894735
    LP_LEBENSPHASE_FEIN
    [0.0]
    The percentage of missing value for LP_LEBENSPHASE_FEIN is 0.10954858559212585
    LP_LEBENSPHASE_GROB
    [0.0]
    The percentage of missing value for LP_LEBENSPHASE_GROB is 0.10611509378706292
    LP_FAMILIE_FEIN
    [0.0]
    The percentage of missing value for LP_FAMILIE_FEIN is 0.08728699166648901
    LP_FAMILIE_GROB
    [0.0]
    The percentage of missing value for LP_FAMILIE_GROB is 0.08728699166648901
    LP_STATUS_FEIN
    [0.0]
    The percentage of missing value for LP_STATUS_FEIN is 0.005446460529992
    LP_STATUS_GROB
    [0.0]
    The percentage of missing value for LP_STATUS_GROB is 0.005446460529992
    NATIONALITAET_KZ
    [-1.0, 0.0]
    The percentage of missing value for NATIONALITAET_KZ is 0.12153551139391913
    PRAEGENDE_JUGENDJAHRE
    [-1.0, 0.0]
    The percentage of missing value for PRAEGENDE_JUGENDJAHRE is 0.12136608091595687
    RETOURTYP_BK_S
    [0.0]
    The percentage of missing value for RETOURTYP_BK_S is 0.005446460529992
    SEMIO_SOZ
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_SOZ is 0.0
    SEMIO_FAM
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_FAM is 0.0
    SEMIO_REL
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_REL is 0.0
    SEMIO_MAT
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_MAT is 0.0
    SEMIO_VERT
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_VERT is 0.0
    SEMIO_LUST
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_LUST is 0.0
    SEMIO_ERL
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_ERL is 0.0
    SEMIO_KULT
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_KULT is 0.0
    SEMIO_RAT
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_RAT is 0.0
    SEMIO_KRIT
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_KRIT is 0.0
    SEMIO_DOM
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_DOM is 0.0
    SEMIO_KAEM
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_KAEM is 0.0
    SEMIO_PFLICHT
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_PFLICHT is 0.0
    SEMIO_TRADV
    [-1.0, 9.0]
    The percentage of missing value for SEMIO_TRADV is 0.0
    SHOPPER_TYP
    [-1.0]
    The percentage of missing value for SHOPPER_TYP is 0.12476815514894735
    SOHO_KZ
    [-1.0]
    The percentage of missing value for SOHO_KZ is 0.08247000463409188
    TITEL_KZ
    [-1.0, 0.0]
    The percentage of missing value for TITEL_KZ is 0.9975763587258379
    VERS_TYP
    [-1.0]
    The percentage of missing value for VERS_TYP is 0.12476815514894735
    ZABEOTYP
    [-1.0, 9.0]
    The percentage of missing value for ZABEOTYP is 0.0
    ALTER_HH
    [0.0]
    The percentage of missing value for ALTER_HH is 0.34813699407890975
    ANZ_PERSONEN
    ['']
    The percentage of missing value for ANZ_PERSONEN is 0.08247000463409188
    ANZ_TITEL
    ['']
    The percentage of missing value for ANZ_TITEL is 0.08247000463409188
    HH_EINKOMMEN_SCORE
    [-1.0, 0.0]
    The percentage of missing value for HH_EINKOMMEN_SCORE is 0.020587486156632306
    KK_KUNDENTYP
    [-1.0]
    The percentage of missing value for KK_KUNDENTYP is 0.6559674873011295
    W_KEIT_KIND_HH
    [-1.0, 0.0]
    The percentage of missing value for W_KEIT_KIND_HH is 0.16605084485217472
    WOHNDAUER_2008
    [-1.0, 0.0]
    The percentage of missing value for WOHNDAUER_2008 is 0.08247000463409188
    ANZ_HAUSHALTE_AKTIV
    [0.0]
    The percentage of missing value for ANZ_HAUSHALTE_AKTIV is 0.11176913470396231
    ANZ_HH_TITEL
    ['']
    The percentage of missing value for ANZ_HH_TITEL is 0.10884842255736793
    GEBAEUDETYP
    [-1.0, 0.0]
    The percentage of missing value for GEBAEUDETYP is 0.10451728583594866
    KONSUMNAEHE
    ['']
    The percentage of missing value for KONSUMNAEHE is 0.08299737102245122
    MIN_GEBAEUDEJAHR
    [0.0]
    The percentage of missing value for MIN_GEBAEUDEJAHR is 0.10451728583594866
    OST_WEST_KZ
    ['-1']
    The percentage of missing value for OST_WEST_KZ is 0.10451728583594866
    WOHNLAGE
    [-1.0]
    The percentage of missing value for WOHNLAGE is 0.10451728583594866
    CAMEO_DEUG_2015
    ['-1', 'X']
    The percentage of missing value for CAMEO_DEUG_2015 is 0.11147852216229195
    CAMEO_DEU_2015
    ['XX']
    The percentage of missing value for CAMEO_DEU_2015 is 0.11147852216229195
    CAMEO_INTL_2015
    ['-1', 'XX']
    The percentage of missing value for CAMEO_INTL_2015 is 0.11147852216229195
    KBA05_ANTG1
    [-1.0]
    The percentage of missing value for KBA05_ANTG1 is 0.14959701353536328
    KBA05_ANTG2
    [-1.0]
    The percentage of missing value for KBA05_ANTG2 is 0.14959701353536328
    KBA05_ANTG3
    [-1.0]
    The percentage of missing value for KBA05_ANTG3 is 0.14959701353536328
    KBA05_ANTG4
    [-1.0]
    The percentage of missing value for KBA05_ANTG4 is 0.14959701353536328
    KBA05_BAUMAX
    [-1.0, 0.0]
    The percentage of missing value for KBA05_BAUMAX is 0.5346866826522265
    KBA05_GBZ
    [-1.0, 0.0]
    The percentage of missing value for KBA05_GBZ is 0.14959701353536328
    BALLRAUM
    [-1.0]
    The percentage of missing value for BALLRAUM is 0.10518154307405234
    EWDICHTE
    [-1.0]
    The percentage of missing value for EWDICHTE is 0.10518154307405234
    INNENSTADT
    [-1.0]
    The percentage of missing value for INNENSTADT is 0.10518154307405234
    GEBAEUDETYP_RASTER
    ['']
    The percentage of missing value for GEBAEUDETYP_RASTER is 0.10452514022896678
    KKK
    [-1.0, 0.0]
    The percentage of missing value for KKK is 0.17735668257368262
    MOBI_REGIO
    ['']
    The percentage of missing value for MOBI_REGIO is 0.14959701353536328
    ONLINE_AFFINITAET
    ['']
    The percentage of missing value for ONLINE_AFFINITAET is 0.005446460529992
    REGIOTYP
    [-1.0, 0.0]
    The percentage of missing value for REGIOTYP is 0.17735668257368262
    KBA13_ANZAHL_PKW
    ['']
    The percentage of missing value for KBA13_ANZAHL_PKW is 0.11871354018812394
    PLZ8_ANTG1
    [-1.0]
    The percentage of missing value for PLZ8_ANTG1 is 0.13073637178657146
    PLZ8_ANTG2
    [-1.0]
    The percentage of missing value for PLZ8_ANTG2 is 0.13073637178657146
    PLZ8_ANTG3
    [-1.0]
    The percentage of missing value for PLZ8_ANTG3 is 0.13073637178657146
    PLZ8_ANTG4
    [-1.0]
    The percentage of missing value for PLZ8_ANTG4 is 0.13073637178657146
    PLZ8_BAUMAX
    [-1.0, 0.0]
    The percentage of missing value for PLZ8_BAUMAX is 0.13073637178657146
    PLZ8_HHZ
    [-1.0]
    The percentage of missing value for PLZ8_HHZ is 0.13073637178657146
    PLZ8_GBZ
    [-1.0]
    The percentage of missing value for PLZ8_GBZ is 0.13073637178657146
    ARBEIT
    [-1.0, 9.0]
    The percentage of missing value for ARBEIT is 0.10926021716274639
    ORTSGR_KLS9
    [-1.0, 0.0]
    The percentage of missing value for ORTSGR_KLS9 is 0.1091468894920564
    RELAT_AB
    [-1.0, 9.0]
    The percentage of missing value for RELAT_AB is 0.10926021716274639



```python
# azdias_filled.to_csv('azdias_filled_nan.csv',index=False)  
```

#### Step 1.1.2: Assess Missing Data in Each Column

How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)

For the remaining features, are there any patterns in which columns have, or share, missing data?


```python
# return te columns list andcorresponding percentages list for nan value percentage in each column larger than n
# by columns

# this will take a very long time to run!
# Params:
#     df: dataframe
#     n: percentage in decimal
# Return: the list of names and percentage of nan values column which large than n

def get_percentage_missing_in_column(df, n):
    name_list = []
    percentage_list = []
    
    for column_name in df:
        column = df[column_name]
        percentage = column.isnull().sum() / len(column)
        if percentage > n:
            name_list.append(column_name)
            percentage_list.append(percentage)
            
    return name_list, percentage_list
```


```python
# azdias_filled_nan = pd.read_csv('azdias_filled_nan.csv')
azdias_filled_nan = azdias_filled
```


```python
# Perform an assessment of how much missing data there is in each column of the
# dataset.

# get the columns that has nan value large than 10%
name_list_10, percentage_list_10 = get_percentage_missing_in_column(azdias_filled_nan, 0.1)

```


```python
# plot the figure
plt.figure(figsize=(30,20))
plt.rcParams.update({'font.size': 22})
plt.barh(name_list_10, width = percentage_list_10)
```




    <BarContainer object of 46 artists>




![png](output_18_1.png)



```python
azdias_filled_nan.shape
```




    (891221, 85)



# Investigate patterns in the amount of missing data in each column.



```python
azdias_filled_nan.head(5)
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
      <th>AGER_TYP</th>
      <th>ALTERSKATEGORIE_GROB</th>
      <th>ANREDE_KZ</th>
      <th>CJT_GESAMTTYP</th>
      <th>FINANZ_MINIMALIST</th>
      <th>FINANZ_SPARER</th>
      <th>FINANZ_VORSORGER</th>
      <th>FINANZ_ANLEGER</th>
      <th>FINANZ_UNAUFFAELLIGER</th>
      <th>FINANZ_HAUSBAUER</th>
      <th>...</th>
      <th>PLZ8_ANTG1</th>
      <th>PLZ8_ANTG2</th>
      <th>PLZ8_ANTG3</th>
      <th>PLZ8_ANTG4</th>
      <th>PLZ8_BAUMAX</th>
      <th>PLZ8_HHZ</th>
      <th>PLZ8_GBZ</th>
      <th>ARBEIT</th>
      <th>ORTSGR_KLS9</th>
      <th>RELAT_AB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>2</td>
      <td>5.0</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>2</td>
      <td>3.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>2.0</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>1</td>
      <td>5.0</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>




```python
# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
print(azdias_filled_nan.shape)
# as we can see from above fligure, we will remove outliers that has missing values more than 20%
name_list_20, percentage_list_20 = get_percentage_missing_in_column(azdias_filled_nan, 0.2)
azdias_filled_nan_dropped = azdias_filled_nan.drop(name_list_20, axis = 1)
```

    (891221, 85)



```python
print(azdias_filled_nan_dropped.shape)
```

    (891221, 79)


#### Discussion 1.1.2: Assess Missing Data in Each Column

As we can see that most of the columns does not appear in the above figure, which means that they have nan value less than 10%.

And we have two columns `KK_KUNDENTYP` and `AGER_TYP` that has much higher missing values, so we removed this two columns from the dataset above

#### Step 1.1.3: Assess Missing Data in Each Row

Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.

In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
- You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
- To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.

Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**


```python
# How much data is missing in each row of the dataset?

# we have 891221 rows, so I only print out first 300, so we could 

for i in range(300):
    nan_count = pd.isnull(azdias_filled_nan.iloc[i].values).sum()
    print(nan_count)
```

    49
    4
    4
    9
    3
    2
    5
    4
    4
    3
    3
    53
    10
    11
    53
    11
    9
    53
    9
    5
    16
    6
    6
    11
    53
    9
    24
    5
    4
    4
    40
    5
    3
    3
    6
    49
    5
    2
    5
    7
    10
    2
    4
    3
    4
    2
    46
    2
    53
    5
    5
    2
    3
    49
    53
    6
    3
    4
    2
    5
    8
    49
    49
    3
    8
    6
    2
    2
    2
    53
    10
    5
    7
    12
    4
    53
    49
    4
    5
    2
    6
    51
    2
    49
    1
    5
    2
    3
    11
    7
    40
    3
    5
    3
    9
    6
    9
    53
    2
    53
    3
    3
    5
    53
    3
    2
    49
    3
    53
    49
    2
    5
    53
    2
    6
    1
    4
    3
    11
    3
    4
    2
    2
    5
    4
    4
    4
    2
    3
    5
    7
    2
    3
    53
    5
    2
    7
    53
    33
    5
    3
    9
    21
    23
    53
    6
    3
    4
    4
    3
    3
    2
    4
    49
    53
    6
    14
    11
    22
    12
    2
    4
    9
    53
    6
    38
    6
    2
    53
    2
    3
    2
    2
    23
    5
    53
    20
    6
    22
    49
    3
    4
    5
    6
    14
    6
    9
    3
    3
    2
    5
    14
    4
    12
    3
    6
    7
    9
    4
    39
    2
    3
    13
    3
    3
    49
    4
    1
    8
    53
    53
    3
    3
    7
    4
    3
    2
    3
    3
    2
    3
    2
    3
    4
    3
    4
    5
    53
    53
    39
    53
    4
    5
    49
    5
    6
    53
    49
    3
    53
    4
    2
    5
    2
    49
    7
    3
    4
    53
    3
    53
    2
    6
    3
    8
    12
    2
    3
    10
    3
    4
    5
    3
    2
    5
    11
    3
    7
    3
    18
    3
    5
    13
    3
    5
    2
    5
    3
    4
    3
    11
    4
    1
    2
    3
    2
    37
    3
    37
    3
    5
    13
    4
    2
    6
    2
    5
    6
    2
    5


As we can see above, most of them are below 10, but there is a high percentage that the total missing values in each rows will be more than 30, so we could seperate the data into two subsets by the wether the missing values are less than 30


```python
# test

# a = np.array([np.nan, 2, 1])
# aa = pd.DataFrame()
# np.nansum(2)
# b = pd.DataFrame([a],columns=list('abq'))
# # b.append(pd.DataFrame(a,columns=['a','b','q']))
# # list(azdias_filled_nan.columns.values)
# aa = aa.append(b, ignore_index = True)
# aa= aa.append(b, ignore_index = True)
# aa
azdias_filled_nan.iloc[[0,2]]
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
      <th>AGER_TYP</th>
      <th>ALTERSKATEGORIE_GROB</th>
      <th>ANREDE_KZ</th>
      <th>CJT_GESAMTTYP</th>
      <th>FINANZ_MINIMALIST</th>
      <th>FINANZ_SPARER</th>
      <th>FINANZ_VORSORGER</th>
      <th>FINANZ_ANLEGER</th>
      <th>FINANZ_UNAUFFAELLIGER</th>
      <th>FINANZ_HAUSBAUER</th>
      <th>...</th>
      <th>PLZ8_ANTG1</th>
      <th>PLZ8_ANTG2</th>
      <th>PLZ8_ANTG3</th>
      <th>PLZ8_ANTG4</th>
      <th>PLZ8_BAUMAX</th>
      <th>PLZ8_HHZ</th>
      <th>PLZ8_GBZ</th>
      <th>ARBEIT</th>
      <th>ORTSGR_KLS9</th>
      <th>RELAT_AB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>1</td>
      <td>2.0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>2</td>
      <td>3.0</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 85 columns</p>
</div>




```python
# Write code to divide the data into two subsets based on the number of missing
# values in each row.
# by rows

# this will take a very long time to run!
# Params:
#     df: dataframe
#     n: threshold
# Return: list of index below threshold and above threshold

def get_misssing_info_in_row(df, n):
    subset_below_threshold_indexes = []
    subset_above_threshold_indexes = []
#     column_names = list(df.columns.values)
    rows_count = df.shape[0]
    print_every = rows_count // 10
    
    for index in range(rows_count):
        row_values = df.iloc[index].values
        nan_count = pd.isnull(row_values).sum()
        if nan_count > n:
            subset_above_threshold_indexes.append(index)
        else:
            subset_below_threshold_indexes.append(index)
        if index % print_every == 0:
            print('{} %'.format(index*10/print_every))
    print('Done!')
    return subset_below_threshold_indexes, subset_above_threshold_indexes

subset_below_threshold_indexes, subset_above_threshold_indexes = get_misssing_info_in_row(azdias, 30)
```

    0.0 %
    10.0 %
    20.0 %
    30.0 %
    40.0 %
    50.0 %
    60.0 %
    70.0 %
    80.0 %
    90.0 %
    100.0 %
    Done!



```python

print('percentage of rows with a lot of missing data: {0:.2f} %'.format(len(subset_above_threshold_indexes)*100/azdias.shape[0]))


```

    percentage of rows with a lot of missing data: 10.46 %



```python
# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

def compare_and_plot(label, data1, data2):
    plt.figure(figsize=(20, 10));  
    plt.subplot(2,2,1)
    sns.countplot(x=label, data=data1)
    plt.subplot(2,2,2)
    sns.countplot(x=label, data=data2)

compare_labels = ['ALTERSKATEGORIE_GROB','ANREDE_KZ','FINANZ_MINIMALIST','FINANZ_SPARER','FINANZ_HAUSBAUER']

```


```python
# get the subsets from indexes,
# original data
subset_below_threshold = azdias.iloc[subset_below_threshold_indexes]
subset_above_threshold = azdias.iloc[subset_above_threshold_indexes]

for item in compare_labels:
    compare_and_plot(item, subset_below_threshold, subset_above_threshold)
```


![png](output_32_0.png)



![png](output_32_1.png)



![png](output_32_2.png)



![png](output_32_3.png)



![png](output_32_4.png)



```python
# get the subsets from indexes,
# filled data with nan filled and drop 2 columns
filled_subset_below_threshold = azdias_filled_nan_dropped.iloc[subset_below_threshold_indexes]
filled_subset_above_threshold = azdias_filled_nan_dropped.iloc[subset_above_threshold_indexes]

for item in compare_labels:
    compare_and_plot(item, filled_subset_below_threshold, filled_subset_above_threshold)
```


![png](output_33_0.png)



![png](output_33_1.png)



![png](output_33_2.png)



![png](output_33_3.png)



![png](output_33_4.png)


#### Discussion 1.1.3: Assess Missing Data in Each Row

(Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)

A: 

As we can see from the above figures, the two subsets do different from each other a lot except for `ANREDE_KZ`. For other cloumns, they have similiar pattern: in the subset with a lot of missing values, most of the data values goes to near 0 while one of the data value goes higher to almoast 90% of this column. 

And we only have 10.45% of the rows that with a lot of missing data.

Based on the above analysis, I think we should remove those rows with a lot of missing data.

### Step 1.2: Select and Re-Encode Features

Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
- For numeric and interval data, these features can be kept without changes.
- Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
- Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.

In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.

Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!


```python
# How many features are there of each data type?
dic = {}
for item in feat_info['type']:
    dic[item] = dic[item] + 1 if item in dic else 1

print(dic)

# show in figure
plt.figure(figsize=(20,6))
sns.countplot(x='type', data=feat_info)
plt.show()

```

    {'categorical': 21, 'ordinal': 49, 'numeric': 7, 'mixed': 7, 'interval': 1}



![png](output_36_1.png)


#### Step 1.2.1: Re-Encode Categorical Features

For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
- For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
- There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
- For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.


```python
# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
object_variable = []
binary_variable = []
large_level_variables = []
small_level_variables = []

for index in range(feat_info.shape[0]):
    type = feat_info['type'][index]
    
    if type == 'categorical':
        attribute = feat_info['attribute'][index]
        if attribute in filled_subset_below_threshold.columns:
            dimensions = filled_subset_below_threshold[attribute].nunique()
            
            print(attribute, dimensions)
            if dimensions == 2:
                if filled_subset_below_threshold[attribute].dtype == 'O':
                    object_variable.append(attribute)
                else:
                    binary_variable.append(attribute)
            else:
                if dimensions > 7:
                    large_level_variables.append(attribute)
                else:
                    small_level_variables.append(attribute)

print('\n')
print('{} binary variables: \n{}\n'.format(len(binary_variable), binary_variable))
print('{} object_variable: \n{}\n'.format(len(object_variable), object_variable))
print('{} small-level variables: \n{}\n'.format(len(small_level_variables), small_level_variables))
print('{} large_level_variables: \n{}\n'.format(len(large_level_variables), large_level_variables))
```

    ANREDE_KZ 2
    CJT_GESAMTTYP 6
    FINANZTYP 6
    GFK_URLAUBERTYP 12
    GREEN_AVANTGARDE 2
    LP_FAMILIE_FEIN 11
    LP_FAMILIE_GROB 5
    LP_STATUS_FEIN 10
    LP_STATUS_GROB 5
    NATIONALITAET_KZ 3
    SHOPPER_TYP 4
    SOHO_KZ 2
    VERS_TYP 2
    ZABEOTYP 6
    GEBAEUDETYP 7
    OST_WEST_KZ 2
    CAMEO_DEUG_2015 9
    CAMEO_DEU_2015 44
    
    
    4 binary variables: 
    ['ANREDE_KZ', 'GREEN_AVANTGARDE', 'SOHO_KZ', 'VERS_TYP']
    
    1 object_variable: 
    ['OST_WEST_KZ']
    
    8 small-level variables: 
    ['CJT_GESAMTTYP', 'FINANZTYP', 'LP_FAMILIE_GROB', 'LP_STATUS_GROB', 'NATIONALITAET_KZ', 'SHOPPER_TYP', 'ZABEOTYP', 'GEBAEUDETYP']
    
    5 large_level_variables: 
    ['GFK_URLAUBERTYP', 'LP_FAMILIE_FEIN', 'LP_STATUS_FEIN', 'CAMEO_DEUG_2015', 'CAMEO_DEU_2015']
    



```python
# drop columns:
print('shape before drop: ', filled_subset_below_threshold.shape)
filled_subset_below_threshold = filled_subset_below_threshold.drop(large_level_variables, axis=1)
print('shape after drop: ', filled_subset_below_threshold.shape)

```

    shape before drop:  (797981, 79)
    shape after drop:  (797981, 74)



```python
# Re-encode categorical variable(s) to be kept in the analysis.

one_hot_data = filled_subset_below_threshold

for item in small_level_variables:
    one_hot_data = pd.concat([one_hot_data, pd.get_dummies(one_hot_data[item], prefix=item)], axis=1)
    one_hot_data = one_hot_data.drop(item, axis=1)


one_hot_data.shape

```




    (797981, 108)




```python
one_hot_data['OST_WEST_KZ'] = one_hot_data['OST_WEST_KZ'].replace(['O','W'], [0,1])

```

## Discussion 1.2.1: Re-Encode Categorical Features

(Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?) 

A:

as we can see, we have 4 binary variables, and 13 multi-level variables, in which 5 of them have more than 7 different values, so I dropped those columns, and encoded the other 8 columns

special treatment was applied to `OST_WEST_KZ`, since it only has 2 dimensions but value is not numerical, I converted it to binary values

#### Step 1.2.2: Engineer Mixed-Type Features

There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
- "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
- "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
- If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.

Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.


```python
# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
# after observe the Data_Dictionary.md, we could get:
print(one_hot_data.shape)

criteria = [one_hot_data['PRAEGENDE_JUGENDJAHRE'].between(1, 2),
            one_hot_data['PRAEGENDE_JUGENDJAHRE'].between(3, 4),
            one_hot_data['PRAEGENDE_JUGENDJAHRE'].between(5, 7),
            one_hot_data['PRAEGENDE_JUGENDJAHRE'].between(8, 9),
            one_hot_data['PRAEGENDE_JUGENDJAHRE'].between(10, 13),
            one_hot_data['PRAEGENDE_JUGENDJAHRE'].between(14, 15)]

values = [1, 2, 3, 4, 5, 6] # one hot encode this???

one_hot_data['PRAEGENDE_JUGENDJAHRE_DECADE'] = np.select(criteria, values, np.nan)
print(one_hot_data['PRAEGENDE_JUGENDJAHRE_DECADE'].head())
# create new PRAEGENDE_JUGENDJAHRE_MOVEMENT
# 0 for mainstream, 1 for avantgrade
mainstream_labels = [1, 3, 5, 8, 10, 12, 14]
one_hot_data['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = one_hot_data['PRAEGENDE_JUGENDJAHRE'].isin(mainstream_labels).astype(int)
print(one_hot_data['PRAEGENDE_JUGENDJAHRE_MOVEMENT'].head())
# drop original column
one_hot_data = one_hot_data.drop('PRAEGENDE_JUGENDJAHRE', axis=1)
print(one_hot_data.shape)


```

    (797981, 108)
    1    6.0
    2    6.0
    3    4.0
    4    4.0
    5    2.0
    Name: PRAEGENDE_JUGENDJAHRE_DECADE, dtype: float64
    1    1
    2    0
    3    1
    4    1
    5    1
    Name: PRAEGENDE_JUGENDJAHRE_MOVEMENT, dtype: int64
    (797981, 109)



```python
# Investigate "CAMEO_INTL_2015" and engineer two new variables.
print(one_hot_data.shape)

CAMEO_INTL_2015 = one_hot_data['CAMEO_INTL_2015'].replace('XX', '0').astype(float)
# wealth
criteria1 = [CAMEO_INTL_2015.between(11, 15),
            CAMEO_INTL_2015.between(21, 25),
            CAMEO_INTL_2015.between(31, 35),
            CAMEO_INTL_2015.between(41, 45),
            CAMEO_INTL_2015.between(51, 55)]

values = [1, 2, 3, 4, 5]
one_hot_data['CAMEO_INTL_2015_WEALTH'] = np.select(criteria1, values, np.nan)
print(one_hot_data['CAMEO_INTL_2015_WEALTH'].head())
# life stage
life_stage1 = [11,21,31,41,51]
life_stage2 = [12,22,32,42,52]
life_stage3 = [13,23,33,43,53]
life_stage4 = [14,24,34,44,54]
life_stage5 = [15,25,35,45,55]

criteria2 = [CAMEO_INTL_2015.isin(life_stage1),
            CAMEO_INTL_2015.isin(life_stage2),
            CAMEO_INTL_2015.isin(life_stage3),
            CAMEO_INTL_2015.isin(life_stage4),
            CAMEO_INTL_2015.isin(life_stage5)]
one_hot_data['CAMEO_INTL_2015_LIFE_STAGE'] = np.select(criteria2, values, np.nan)
print(one_hot_data['CAMEO_INTL_2015_LIFE_STAGE'].head())

one_hot_data = one_hot_data.drop('CAMEO_INTL_2015', axis=1)

print(one_hot_data.shape)


```

    (797981, 109)
    1    5.0
    2    2.0
    3    1.0
    4    4.0
    5    5.0
    Name: CAMEO_INTL_2015_WEALTH, dtype: float64
    1    1.0
    2    4.0
    3    2.0
    4    3.0
    5    4.0
    Name: CAMEO_INTL_2015_LIFE_STAGE, dtype: float64
    (797981, 110)


#### Discussion 1.2.2: Engineer Mixed-Type Features

(Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
A:

for `PRAEGENDE_JUGENDJAHRE`, I divided it into two columns, one by DECADE, and one by MOVEMENT,

and for `CAMEO_INTL_2015`, I divided it into two columns, one by WEALTH, and one by LIFE STAGE,

at this time, I am only making two columns changes, and will investgating more in the future


#### Step 1.2.3: Complete Feature Selection

In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
- All numeric, interval, and ordinal type columns from the original dataset.
- Binary categorical features (all numerically-encoded).
- Engineered features from other multi-level categorical features and mixed features.

Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.


```python
# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)

```


```python
# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

# we gonna check the data by columnsagain, to make sure our data is correct:

#column
name_list_10, percentage_list_10 = get_percentage_missing_in_column(one_hot_data, 0)
```


```python
# column, plot the figure:

plt.figure(figsize=(30,20))
plt.rcParams.update({'font.size': 22})
plt.barh(name_list_10, width = percentage_list_10)
```




    <BarContainer object of 37 artists>




![png](output_50_1.png)


As we can see, that we have removed those that have missing values large than 20%


```python
subset_below_threshold_indexes, subset_above_threshold_indexes = get_misssing_info_in_row(one_hot_data, 30)
print('percentage of rows with a lot of missing data: {0:.2f} %'.format(len(subset_above_threshold_indexes)*100/one_hot_data.shape[0]))


```

    0.0 %
    10.0 %
    20.0 %
    30.0 %
    40.0 %
    50.0 %
    60.0 %
    70.0 %
    80.0 %
    90.0 %
    100.0 %
    Done!
    percentage of rows with a lot of missing data: 0.00 %



```python
# check that we have dropped the columns with large_level_variables:

for item in large_level_variables:
    print(item not in one_hot_data.columns.values)
```

    True
    True
    True
    True
    True



```python
# check we have encoded small_level_variables and removed the original columns
for item in small_level_variables:
    # check we removed the original column
    print(item not in one_hot_data.columns.values)
    # check we created new columns based on the origional column
    print(item + '_' in ''.join(one_hot_data.columns.values))

```

    True
    True
    True
    True
    True
    True
    True
    True
    True
    True
    True
    True
    True
    True
    True
    True



```python
# check that we have dropped the two mixed type features, and created Engineered features columns based on that 
engineered_feature = ['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015']

for item in engineered_feature:
    # check we removed the original column
    print(item not in one_hot_data.columns.values)
    # check we created new columns based on the origional column
    print(item + '_' in ''.join(one_hot_data.columns.values))
```

    True
    True
    True
    True



```python
one_hot_data.shape
```




    (797981, 110)



### Step 1.3: Create a Cleaning Function

Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.


```python
def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    print('convert missing value codes into NaNs...')
    for index, row in feat_info.iterrows():
        column_name = row['attribute']
        missing_or_unknow_list = row['missing_or_unknown'][1:-1].split(',')
        df[column_name] = df[column_name].replace(missing_or_unknow_list, np.nan)
    
    print('df shape {}'.format(df.shape))
    # remove selected columns and rows, ...
    print('drop columns with more than 20% missing values')
    name_list_20, percentage_list_20 = get_percentage_missing_in_column(df, 0.2)
    df = df.drop(name_list_20, axis = 1)
    
    print('df shape {}'.format(df.shape))
    print('selecting rows that has less than 30% of missing values..')
    df, subset_above_threshold_indexes = get_misssing_info_in_row(df, 30)
    print('df shape {}'.format(df.shape))
    
    # select, re-encode, and engineer column values.
    # Re-encode categorical variable(s) to be kept in the analysis.
    # drop columns:
    large_level_variables = []
    small_level_variables = []
    print('investgating the categorical variables...')
    
    for index in range(feat_info.shape[0]):
        type = feat_info['type'][index]

        if type == 'categorical':
            attribute = feat_info['attribute'][index]
            if attribute in df.columns:
                dimensions = df[attribute].nunique()

                if dimensions > 7:
                    large_level_variables.append(attribute)
                else:
                    small_level_variables.append(attribute)
    print('drop columns with categorical type that has more than 7 different values')
    df = df.drop(large_level_variables, axis=1)
    print('df shape {}'.format(df.shape))
    
    print('one-hot encoding...')
    for item in small_level_variables:
        df = pd.concat([df, pd.get_dummies(df[item], prefix=item)], axis=1)
        df = df.drop(item, axis=1)
    
    # special treatment to OST_WEST_KZ
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace(['O','W'], [0,1])

    print('df shape {}'.format(df.shape))
    
    # engineeded:
    
    # after observe the Data_Dictionary.md, we could get:
    print('investgating engineered features')
    criteria = [df['PRAEGENDE_JUGENDJAHRE'].between(1, 2),
                df['PRAEGENDE_JUGENDJAHRE'].between(3, 4),
                df['PRAEGENDE_JUGENDJAHRE'].between(5, 7),
                df['PRAEGENDE_JUGENDJAHRE'].between(8, 9),
                df['PRAEGENDE_JUGENDJAHRE'].between(10, 13),
                df['PRAEGENDE_JUGENDJAHRE'].between(14, 15)]

    values = [1, 2, 3, 4, 5, 6] # one hot encode this???

    df['PRAEGENDE_JUGENDJAHRE_DECADE'] = np.select(criteria, values, np.nan)
    # create new PRAEGENDE_JUGENDJAHRE_MOVEMENT
    # 0 for mainstream, 1 for avantgrade
    mainstream_labels = [1, 3, 5, 8, 10, 12, 14]
    df['PRAEGENDE_JUGENDJAHRE_MOVEMENT'] = df['PRAEGENDE_JUGENDJAHRE'].isin(mainstream_labels).astype(int)
    # drop original column
    df = df.drop('PRAEGENDE_JUGENDJAHRE', axis=1)
    
    

    CAMEO_INTL_2015 = df['CAMEO_INTL_2015'].replace('XX', '0').astype(float)
    # wealth
    criteria1 = [CAMEO_INTL_2015.between(11, 15),
                CAMEO_INTL_2015.between(21, 25),
                CAMEO_INTL_2015.between(31, 35),
                CAMEO_INTL_2015.between(41, 45),
                CAMEO_INTL_2015.between(51, 55)]

    values1 = [1, 2, 3, 4, 5]
    df['CAMEO_INTL_2015_WEALTH'] = np.select(criteria1, values1, np.nan)
    # life stage
    life_stage1 = [11,21,31,41,51]
    life_stage2 = [12,22,32,42,52]
    life_stage3 = [13,23,33,43,53]
    life_stage4 = [14,24,34,44,54]
    life_stage5 = [15,25,35,45,55]

    criteria2 = [CAMEO_INTL_2015.isin(life_stage1),
                CAMEO_INTL_2015.isin(life_stage2),
                CAMEO_INTL_2015.isin(life_stage3),
                CAMEO_INTL_2015.isin(life_stage4),
                CAMEO_INTL_2015.isin(life_stage5)]
    df['CAMEO_INTL_2015_LIFE_STAGE'] = np.select(criteria2, values1, np.nan)

    df = df.drop('CAMEO_INTL_2015', axis=1)
    print('df shape {}'.format(df.shape))
    print('Done!')
    # Return the cleaned dataframe.
    return df;
    
```

## Step 2: Feature Transformation

### Step 2.1: Apply Feature Scaling

Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:

- sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
- For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
- For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.


```python
# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
def get_none_object_list(df):
    g = df.columns.to_series().groupby(df.dtypes).groups

    # get none object columns
    list1 = []
    for k, v in g.items():
        print(k.name)
        if k.name != 'object':
            list1 = list1 + v.tolist()
    return list1

def replace_missing_values(df):
    print(df.shape)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    list1 = get_none_object_list(df)

    df[list1]  = imp.fit_transform(df[list1])
    print(df.shape)
    return df

one_hot_data = replace_missing_values(one_hot_data)
```

    (797981, 110)
    uint8
    int64
    float64
    (797981, 110)



```python
# test
name_list_10, percentage_list_10 = get_percentage_missing_in_column(one_hot_data, 0)

# we can see that there is no nan values in columns
name_list_10
```




    []




```python
# Apply feature scaling to the general population demographics data.

def feature_scale(df):
    list1 = get_none_object_list(df)
    df[list1] = p.StandardScaler().fit_transform(df[list1]) 
    return df

one_hot_data = feature_scale(one_hot_data)
```

    float64



```python
one_hot_data.head()
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
      <th>ALTERSKATEGORIE_GROB</th>
      <th>ANREDE_KZ</th>
      <th>FINANZ_MINIMALIST</th>
      <th>FINANZ_SPARER</th>
      <th>FINANZ_VORSORGER</th>
      <th>FINANZ_ANLEGER</th>
      <th>FINANZ_UNAUFFAELLIGER</th>
      <th>FINANZ_HAUSBAUER</th>
      <th>GREEN_AVANTGARDE</th>
      <th>HEALTH_TYP</th>
      <th>...</th>
      <th>GEBAEUDETYP_2.0</th>
      <th>GEBAEUDETYP_3.0</th>
      <th>GEBAEUDETYP_4.0</th>
      <th>GEBAEUDETYP_5.0</th>
      <th>GEBAEUDETYP_6.0</th>
      <th>GEBAEUDETYP_8.0</th>
      <th>PRAEGENDE_JUGENDJAHRE_DECADE</th>
      <th>PRAEGENDE_JUGENDJAHRE_MOVEMENT</th>
      <th>CAMEO_INTL_2015_WEALTH</th>
      <th>CAMEO_INTL_2015_LIFE_STAGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-1.766017</td>
      <td>0.957942</td>
      <td>-1.494612</td>
      <td>1.537983</td>
      <td>-1.040686</td>
      <td>1.466053</td>
      <td>0.958791</td>
      <td>1.339250</td>
      <td>-0.530436</td>
      <td>1.085897</td>
      <td>...</td>
      <td>-0.0787</td>
      <td>-0.537083</td>
      <td>-0.033584</td>
      <td>-0.001119</td>
      <td>-0.02802</td>
      <td>2.057585</td>
      <td>1.165655</td>
      <td>0.585899</td>
      <td>1.191260</td>
      <td>-1.266372</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.201260</td>
      <td>0.957942</td>
      <td>-1.494612</td>
      <td>0.864615</td>
      <td>-1.766961</td>
      <td>-0.570962</td>
      <td>0.244236</td>
      <td>1.339250</td>
      <td>1.885243</td>
      <td>1.085897</td>
      <td>...</td>
      <td>-0.0787</td>
      <td>-0.537083</td>
      <td>-0.033584</td>
      <td>-0.001119</td>
      <td>-0.02802</td>
      <td>-0.486007</td>
      <td>1.165655</td>
      <td>-1.706778</td>
      <td>-0.865143</td>
      <td>0.761955</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.184898</td>
      <td>0.957942</td>
      <td>0.683135</td>
      <td>-0.482122</td>
      <td>1.138139</td>
      <td>-0.570962</td>
      <td>-1.184875</td>
      <td>-0.791295</td>
      <td>-0.530436</td>
      <td>-0.269842</td>
      <td>...</td>
      <td>-0.0787</td>
      <td>-0.537083</td>
      <td>-0.033584</td>
      <td>-0.001119</td>
      <td>-0.02802</td>
      <td>-0.486007</td>
      <td>-0.232000</td>
      <td>0.585899</td>
      <td>-1.550611</td>
      <td>-0.590263</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.201260</td>
      <td>-1.043905</td>
      <td>0.683135</td>
      <td>0.191246</td>
      <td>0.411864</td>
      <td>-1.249968</td>
      <td>0.244236</td>
      <td>-0.791295</td>
      <td>-0.530436</td>
      <td>1.085897</td>
      <td>...</td>
      <td>-0.0787</td>
      <td>-0.537083</td>
      <td>-0.033584</td>
      <td>-0.001119</td>
      <td>-0.02802</td>
      <td>-0.486007</td>
      <td>-0.232000</td>
      <td>0.585899</td>
      <td>0.505792</td>
      <td>0.085846</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.766017</td>
      <td>0.957942</td>
      <td>-0.042781</td>
      <td>-1.155491</td>
      <td>1.138139</td>
      <td>-0.570962</td>
      <td>-0.470319</td>
      <td>1.339250</td>
      <td>-0.530436</td>
      <td>1.085897</td>
      <td>...</td>
      <td>-0.0787</td>
      <td>-0.537083</td>
      <td>-0.033584</td>
      <td>-0.001119</td>
      <td>-0.02802</td>
      <td>-0.486007</td>
      <td>-1.629656</td>
      <td>0.585899</td>
      <td>1.191260</td>
      <td>0.761955</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 110 columns</p>
</div>



### Discussion 2.1: Apply Feature Scaling

(Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)
A:

Applied StandardScaler feature scaling from sklearn to only the none-object type.

### Step 2.2: Perform Dimensionality Reduction

On your scaled data, you are now ready to apply dimensionality reduction techniques.

- Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
- Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
- Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.


```python
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components

    INPUT: pca - the result of instantian of PCA in scikit learn

    OUTPUT: None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


```


```python
# Apply PCA to the data.

pca = PCA()
one_hot_data_pca = pca.fit_transform(one_hot_data)
```


```python
# Investigate the variance accounted for by each principal component.

scree_plot(pca)
```


![png](output_68_0.png)



```python
# Re-apply PCA to the data while selecting for number of components to retain.

pca = PCA(80)
one_hot_data_pca = pca.fit_transform(one_hot_data)
```


```python
one_hot_data_pca.shape
```




    (797981, 80)



### Discussion 2.2: Perform Dimensionality Reduction

(Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)

As we can see from the above figure, with 80 components, it will explain about 95% explained_variance_ratio_, so we gonan use 80

### Step 2.3: Interpret Principal Components

Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.

As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.

- To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
- You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.


```python
# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

def get_format(n):
    n = n + 1
    if n == 1:
        return '1st'
    elif n == 2:
        return '2nd'
    elif n == 3:
        return '3rd'
    else:
        return str(n) + 'th'
    
def get_sorted_name_weights_dic(pca, n):
    df = pd.DataFrame(pca.components_[n])
    df.index = one_hot_data.columns
    column_name = 'weights of {} principal component'.format(get_format(n))
    df.columns = [column_name]
    return df.sort_values(by=[column_name], ascending=False)

get_sorted_name_weights_dic(pca, 0)
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
      <th>weights of 1st principal component</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LP_STATUS_GROB_1.0</th>
      <td>0.199146</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG3</th>
      <td>0.193288</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG4</th>
      <td>0.188476</td>
    </tr>
    <tr>
      <th>HH_EINKOMMEN_SCORE</th>
      <td>0.185999</td>
    </tr>
    <tr>
      <th>PLZ8_BAUMAX</th>
      <td>0.184342</td>
    </tr>
    <tr>
      <th>CAMEO_INTL_2015_WEALTH</th>
      <td>0.179197</td>
    </tr>
    <tr>
      <th>ORTSGR_KLS9</th>
      <td>0.168464</td>
    </tr>
    <tr>
      <th>EWDICHTE</th>
      <td>0.166664</td>
    </tr>
    <tr>
      <th>FINANZ_HAUSBAUER</th>
      <td>0.150310</td>
    </tr>
    <tr>
      <th>KBA05_ANTG4</th>
      <td>0.132956</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG2</th>
      <td>0.131750</td>
    </tr>
    <tr>
      <th>FINANZ_SPARER</th>
      <td>0.129032</td>
    </tr>
    <tr>
      <th>FINANZTYP_1</th>
      <td>0.123087</td>
    </tr>
    <tr>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <td>0.119826</td>
    </tr>
    <tr>
      <th>ARBEIT</th>
      <td>0.119287</td>
    </tr>
    <tr>
      <th>KBA05_ANTG3</th>
      <td>0.117854</td>
    </tr>
    <tr>
      <th>RELAT_AB</th>
      <td>0.113317</td>
    </tr>
    <tr>
      <th>SEMIO_PFLICHT</th>
      <td>0.096835</td>
    </tr>
    <tr>
      <th>LP_FAMILIE_GROB_1.0</th>
      <td>0.094797</td>
    </tr>
    <tr>
      <th>SEMIO_REL</th>
      <td>0.092340</td>
    </tr>
    <tr>
      <th>PRAEGENDE_JUGENDJAHRE_DECADE</th>
      <td>0.087147</td>
    </tr>
    <tr>
      <th>PRAEGENDE_JUGENDJAHRE_MOVEMENT</th>
      <td>0.085115</td>
    </tr>
    <tr>
      <th>ZABEOTYP_5</th>
      <td>0.085076</td>
    </tr>
    <tr>
      <th>SEMIO_RAT</th>
      <td>0.081126</td>
    </tr>
    <tr>
      <th>SEMIO_TRADV</th>
      <td>0.073789</td>
    </tr>
    <tr>
      <th>GEBAEUDETYP_3.0</th>
      <td>0.072711</td>
    </tr>
    <tr>
      <th>FINANZ_UNAUFFAELLIGER</th>
      <td>0.064767</td>
    </tr>
    <tr>
      <th>SEMIO_MAT</th>
      <td>0.063422</td>
    </tr>
    <tr>
      <th>FINANZ_ANLEGER</th>
      <td>0.062088</td>
    </tr>
    <tr>
      <th>SEMIO_FAM</th>
      <td>0.060544</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>FINANZTYP_3</th>
      <td>-0.051226</td>
    </tr>
    <tr>
      <th>WOHNDAUER_2008</th>
      <td>-0.056687</td>
    </tr>
    <tr>
      <th>SEMIO_ERL</th>
      <td>-0.058970</td>
    </tr>
    <tr>
      <th>SEMIO_LUST</th>
      <td>-0.060174</td>
    </tr>
    <tr>
      <th>KBA13_ANZAHL_PKW</th>
      <td>-0.065006</td>
    </tr>
    <tr>
      <th>WOHNLAGE</th>
      <td>-0.069836</td>
    </tr>
    <tr>
      <th>NATIONALITAET_KZ_1.0</th>
      <td>-0.071501</td>
    </tr>
    <tr>
      <th>LP_FAMILIE_GROB_5.0</th>
      <td>-0.076301</td>
    </tr>
    <tr>
      <th>FINANZTYP_2</th>
      <td>-0.086611</td>
    </tr>
    <tr>
      <th>ANZ_PERSONEN</th>
      <td>-0.090912</td>
    </tr>
    <tr>
      <th>GEBAEUDETYP_1.0</th>
      <td>-0.095247</td>
    </tr>
    <tr>
      <th>ZABEOTYP_1</th>
      <td>-0.096650</td>
    </tr>
    <tr>
      <th>FINANZ_VORSORGER</th>
      <td>-0.098494</td>
    </tr>
    <tr>
      <th>ALTERSKATEGORIE_GROB</th>
      <td>-0.099887</td>
    </tr>
    <tr>
      <th>GREEN_AVANTGARDE</th>
      <td>-0.100927</td>
    </tr>
    <tr>
      <th>GEBAEUDETYP_RASTER</th>
      <td>-0.105172</td>
    </tr>
    <tr>
      <th>BALLRAUM</th>
      <td>-0.107318</td>
    </tr>
    <tr>
      <th>CAMEO_INTL_2015_LIFE_STAGE</th>
      <td>-0.108549</td>
    </tr>
    <tr>
      <th>LP_STATUS_GROB_5.0</th>
      <td>-0.113988</td>
    </tr>
    <tr>
      <th>LP_STATUS_GROB_4.0</th>
      <td>-0.117623</td>
    </tr>
    <tr>
      <th>LP_LEBENSPHASE_GROB</th>
      <td>-0.128741</td>
    </tr>
    <tr>
      <th>INNENSTADT</th>
      <td>-0.139374</td>
    </tr>
    <tr>
      <th>PLZ8_GBZ</th>
      <td>-0.142044</td>
    </tr>
    <tr>
      <th>LP_LEBENSPHASE_FEIN</th>
      <td>-0.143316</td>
    </tr>
    <tr>
      <th>KONSUMNAEHE</th>
      <td>-0.145795</td>
    </tr>
    <tr>
      <th>KBA05_GBZ</th>
      <td>-0.188553</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG1</th>
      <td>-0.193206</td>
    </tr>
    <tr>
      <th>KBA05_ANTG1</th>
      <td>-0.197480</td>
    </tr>
    <tr>
      <th>FINANZ_MINIMALIST</th>
      <td>-0.201979</td>
    </tr>
    <tr>
      <th>MOBI_REGIO</th>
      <td>-0.211696</td>
    </tr>
  </tbody>
</table>
<p>110 rows × 1 columns</p>
</div>




```python
# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

get_sorted_name_weights_dic(pca, 1)
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
      <th>weights of 2nd principal component</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALTERSKATEGORIE_GROB</th>
      <td>0.235719</td>
    </tr>
    <tr>
      <th>FINANZ_VORSORGER</th>
      <td>0.220082</td>
    </tr>
    <tr>
      <th>ZABEOTYP_3</th>
      <td>0.207098</td>
    </tr>
    <tr>
      <th>SEMIO_ERL</th>
      <td>0.193173</td>
    </tr>
    <tr>
      <th>SEMIO_LUST</th>
      <td>0.165727</td>
    </tr>
    <tr>
      <th>RETOURTYP_BK_S</th>
      <td>0.155718</td>
    </tr>
    <tr>
      <th>W_KEIT_KIND_HH</th>
      <td>0.127412</td>
    </tr>
    <tr>
      <th>CJT_GESAMTTYP_2.0</th>
      <td>0.110429</td>
    </tr>
    <tr>
      <th>FINANZ_HAUSBAUER</th>
      <td>0.102384</td>
    </tr>
    <tr>
      <th>FINANZTYP_5</th>
      <td>0.096051</td>
    </tr>
    <tr>
      <th>FINANZTYP_2</th>
      <td>0.091564</td>
    </tr>
    <tr>
      <th>SEMIO_KRIT</th>
      <td>0.087287</td>
    </tr>
    <tr>
      <th>SHOPPER_TYP_3.0</th>
      <td>0.077749</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG3</th>
      <td>0.077107</td>
    </tr>
    <tr>
      <th>SEMIO_KAEM</th>
      <td>0.075962</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG4</th>
      <td>0.074067</td>
    </tr>
    <tr>
      <th>EWDICHTE</th>
      <td>0.073913</td>
    </tr>
    <tr>
      <th>FINANZ_MINIMALIST</th>
      <td>0.073546</td>
    </tr>
    <tr>
      <th>ORTSGR_KLS9</th>
      <td>0.073529</td>
    </tr>
    <tr>
      <th>CJT_GESAMTTYP_1.0</th>
      <td>0.071663</td>
    </tr>
    <tr>
      <th>PLZ8_BAUMAX</th>
      <td>0.070269</td>
    </tr>
    <tr>
      <th>FINANZTYP_6</th>
      <td>0.067682</td>
    </tr>
    <tr>
      <th>NATIONALITAET_KZ_1.0</th>
      <td>0.064421</td>
    </tr>
    <tr>
      <th>CAMEO_INTL_2015_WEALTH</th>
      <td>0.063768</td>
    </tr>
    <tr>
      <th>LP_FAMILIE_GROB_1.0</th>
      <td>0.063729</td>
    </tr>
    <tr>
      <th>KBA05_ANTG4</th>
      <td>0.057803</td>
    </tr>
    <tr>
      <th>ANREDE_KZ</th>
      <td>0.057265</td>
    </tr>
    <tr>
      <th>ARBEIT</th>
      <td>0.056868</td>
    </tr>
    <tr>
      <th>ANZ_HAUSHALTE_AKTIV</th>
      <td>0.054404</td>
    </tr>
    <tr>
      <th>WOHNDAUER_2008</th>
      <td>0.053050</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>SHOPPER_TYP_0.0</th>
      <td>-0.056755</td>
    </tr>
    <tr>
      <th>CJT_GESAMTTYP_4.0</th>
      <td>-0.057513</td>
    </tr>
    <tr>
      <th>KONSUMNAEHE</th>
      <td>-0.058863</td>
    </tr>
    <tr>
      <th>INNENSTADT</th>
      <td>-0.059213</td>
    </tr>
    <tr>
      <th>PLZ8_GBZ</th>
      <td>-0.060143</td>
    </tr>
    <tr>
      <th>KBA05_ANTG1</th>
      <td>-0.065853</td>
    </tr>
    <tr>
      <th>LP_FAMILIE_GROB_4.0</th>
      <td>-0.066109</td>
    </tr>
    <tr>
      <th>ANZ_PERSONEN</th>
      <td>-0.070774</td>
    </tr>
    <tr>
      <th>ZABEOTYP_1</th>
      <td>-0.071498</td>
    </tr>
    <tr>
      <th>KBA05_GBZ</th>
      <td>-0.072870</td>
    </tr>
    <tr>
      <th>MOBI_REGIO</th>
      <td>-0.074073</td>
    </tr>
    <tr>
      <th>SEMIO_SOZ</th>
      <td>-0.074733</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG1</th>
      <td>-0.075885</td>
    </tr>
    <tr>
      <th>FINANZTYP_3</th>
      <td>-0.082279</td>
    </tr>
    <tr>
      <th>ZABEOTYP_5</th>
      <td>-0.089197</td>
    </tr>
    <tr>
      <th>FINANZTYP_4</th>
      <td>-0.093072</td>
    </tr>
    <tr>
      <th>ZABEOTYP_4</th>
      <td>-0.102913</td>
    </tr>
    <tr>
      <th>FINANZTYP_1</th>
      <td>-0.123810</td>
    </tr>
    <tr>
      <th>SEMIO_MAT</th>
      <td>-0.137692</td>
    </tr>
    <tr>
      <th>SEMIO_FAM</th>
      <td>-0.147841</td>
    </tr>
    <tr>
      <th>SEMIO_RAT</th>
      <td>-0.163756</td>
    </tr>
    <tr>
      <th>ONLINE_AFFINITAET</th>
      <td>-0.166338</td>
    </tr>
    <tr>
      <th>SEMIO_KULT</th>
      <td>-0.179815</td>
    </tr>
    <tr>
      <th>FINANZ_ANLEGER</th>
      <td>-0.197333</td>
    </tr>
    <tr>
      <th>SEMIO_PFLICHT</th>
      <td>-0.207318</td>
    </tr>
    <tr>
      <th>SEMIO_TRADV</th>
      <td>-0.211909</td>
    </tr>
    <tr>
      <th>FINANZ_UNAUFFAELLIGER</th>
      <td>-0.216559</td>
    </tr>
    <tr>
      <th>SEMIO_REL</th>
      <td>-0.222329</td>
    </tr>
    <tr>
      <th>FINANZ_SPARER</th>
      <td>-0.224208</td>
    </tr>
    <tr>
      <th>PRAEGENDE_JUGENDJAHRE_DECADE</th>
      <td>-0.238449</td>
    </tr>
  </tbody>
</table>
<p>110 rows × 1 columns</p>
</div>




```python
# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

get_sorted_name_weights_dic(pca, 2)
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
      <th>weights of 3rd principal component</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SEMIO_VERT</th>
      <td>0.326021</td>
    </tr>
    <tr>
      <th>SEMIO_SOZ</th>
      <td>0.257104</td>
    </tr>
    <tr>
      <th>SEMIO_FAM</th>
      <td>0.256383</td>
    </tr>
    <tr>
      <th>SEMIO_KULT</th>
      <td>0.244876</td>
    </tr>
    <tr>
      <th>FINANZTYP_5</th>
      <td>0.139606</td>
    </tr>
    <tr>
      <th>FINANZ_MINIMALIST</th>
      <td>0.136630</td>
    </tr>
    <tr>
      <th>SHOPPER_TYP_0.0</th>
      <td>0.125754</td>
    </tr>
    <tr>
      <th>ZABEOTYP_1</th>
      <td>0.110223</td>
    </tr>
    <tr>
      <th>SEMIO_REL</th>
      <td>0.098434</td>
    </tr>
    <tr>
      <th>RETOURTYP_BK_S</th>
      <td>0.087432</td>
    </tr>
    <tr>
      <th>W_KEIT_KIND_HH</th>
      <td>0.079010</td>
    </tr>
    <tr>
      <th>SEMIO_MAT</th>
      <td>0.077754</td>
    </tr>
    <tr>
      <th>FINANZ_VORSORGER</th>
      <td>0.068228</td>
    </tr>
    <tr>
      <th>GREEN_AVANTGARDE</th>
      <td>0.055407</td>
    </tr>
    <tr>
      <th>SHOPPER_TYP_1.0</th>
      <td>0.053662</td>
    </tr>
    <tr>
      <th>ORTSGR_KLS9</th>
      <td>0.053376</td>
    </tr>
    <tr>
      <th>EWDICHTE</th>
      <td>0.052722</td>
    </tr>
    <tr>
      <th>PLZ8_BAUMAX</th>
      <td>0.051329</td>
    </tr>
    <tr>
      <th>ZABEOTYP_6</th>
      <td>0.050859</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG4</th>
      <td>0.048327</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG3</th>
      <td>0.047263</td>
    </tr>
    <tr>
      <th>ALTERSKATEGORIE_GROB</th>
      <td>0.037971</td>
    </tr>
    <tr>
      <th>LP_STATUS_GROB_5.0</th>
      <td>0.037375</td>
    </tr>
    <tr>
      <th>SEMIO_LUST</th>
      <td>0.037016</td>
    </tr>
    <tr>
      <th>ARBEIT</th>
      <td>0.033560</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG2</th>
      <td>0.032407</td>
    </tr>
    <tr>
      <th>RELAT_AB</th>
      <td>0.031977</td>
    </tr>
    <tr>
      <th>GEBAEUDETYP_3.0</th>
      <td>0.028236</td>
    </tr>
    <tr>
      <th>CJT_GESAMTTYP_1.0</th>
      <td>0.026000</td>
    </tr>
    <tr>
      <th>LP_STATUS_GROB_3.0</th>
      <td>0.025650</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>WOHNLAGE</th>
      <td>-0.028477</td>
    </tr>
    <tr>
      <th>ZABEOTYP_5</th>
      <td>-0.033457</td>
    </tr>
    <tr>
      <th>NATIONALITAET_KZ_3.0</th>
      <td>-0.034874</td>
    </tr>
    <tr>
      <th>CJT_GESAMTTYP_6.0</th>
      <td>-0.035436</td>
    </tr>
    <tr>
      <th>ONLINE_AFFINITAET</th>
      <td>-0.036263</td>
    </tr>
    <tr>
      <th>PLZ8_GBZ</th>
      <td>-0.037726</td>
    </tr>
    <tr>
      <th>GEBAEUDETYP_RASTER</th>
      <td>-0.037816</td>
    </tr>
    <tr>
      <th>SEMIO_TRADV</th>
      <td>-0.038813</td>
    </tr>
    <tr>
      <th>BALLRAUM</th>
      <td>-0.041014</td>
    </tr>
    <tr>
      <th>KONSUMNAEHE</th>
      <td>-0.043320</td>
    </tr>
    <tr>
      <th>PRAEGENDE_JUGENDJAHRE_MOVEMENT</th>
      <td>-0.043358</td>
    </tr>
    <tr>
      <th>SEMIO_PFLICHT</th>
      <td>-0.043621</td>
    </tr>
    <tr>
      <th>PLZ8_ANTG1</th>
      <td>-0.047465</td>
    </tr>
    <tr>
      <th>INNENSTADT</th>
      <td>-0.048335</td>
    </tr>
    <tr>
      <th>FINANZ_HAUSBAUER</th>
      <td>-0.048559</td>
    </tr>
    <tr>
      <th>SHOPPER_TYP_3.0</th>
      <td>-0.050006</td>
    </tr>
    <tr>
      <th>LP_FAMILIE_GROB_3.0</th>
      <td>-0.050191</td>
    </tr>
    <tr>
      <th>FINANZ_UNAUFFAELLIGER</th>
      <td>-0.068754</td>
    </tr>
    <tr>
      <th>ZABEOTYP_4</th>
      <td>-0.070104</td>
    </tr>
    <tr>
      <th>PRAEGENDE_JUGENDJAHRE_DECADE</th>
      <td>-0.072983</td>
    </tr>
    <tr>
      <th>FINANZ_SPARER</th>
      <td>-0.074752</td>
    </tr>
    <tr>
      <th>SHOPPER_TYP_2.0</th>
      <td>-0.097402</td>
    </tr>
    <tr>
      <th>FINANZTYP_1</th>
      <td>-0.111522</td>
    </tr>
    <tr>
      <th>FINANZ_ANLEGER</th>
      <td>-0.163142</td>
    </tr>
    <tr>
      <th>SEMIO_RAT</th>
      <td>-0.175939</td>
    </tr>
    <tr>
      <th>SEMIO_ERL</th>
      <td>-0.197009</td>
    </tr>
    <tr>
      <th>SEMIO_KRIT</th>
      <td>-0.269879</td>
    </tr>
    <tr>
      <th>SEMIO_DOM</th>
      <td>-0.295520</td>
    </tr>
    <tr>
      <th>SEMIO_KAEM</th>
      <td>-0.323738</td>
    </tr>
    <tr>
      <th>ANREDE_KZ</th>
      <td>-0.352053</td>
    </tr>
  </tbody>
</table>
<p>110 rows × 1 columns</p>
</div>



### Discussion 2.3: Interpret Principal Components

(Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)

A:

1. For the **1st** principal component:

we can see that `low-income earners` has higher positive weights, while `houseowners` and `top earners` has lower negative weight; same thing for `PLZ8_ANTG`, while if you have more than 6 family houses in the PLZ8 region, you get higher weights, but if you have Number of 1-2 family houses in the PLZ8 region, you got very low negative weights. So probably the 1st principal component is about the `wealth` of the family.

2. For the **2nd** principal component:

as we can see that the very top positive element `ALTERSKATEGORIE_GROB` and the very bottom element `PRAEGENDE_JUGENDJAHRE_DECADE` are all about the age of the person, but the second, and third from top and the second from bottom are all about the `typology` in finical, energy, so it's hard to say, probably this component will about both `age` and `typology`

3. For the **3rd** principal component:

most of the elements from both top and bottom are about `Financial typology`




## Step 3: Clustering

### Step 3.1: Apply Clustering to General Population

You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.

- Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
- Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
- Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
- Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.


```python
def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))

    return score

def fit_mods(data, range_list):
    scores = []

    for center in range_list:
        print('applying {} clusters'.format(center))
        scores.append(get_kmeans_score(data, center))

    return scores
```


```python
# Over a number of different cluster counts...


    # run k-means clustering on the data and...
    
    
    # compute the average within-cluster distances.
    
centers = range(5, 40, 5)
scores = fit_mods(one_hot_data_pca, centers)
```

    applying 5 clusters
    applying 10 clusters
    applying 15 clusters
    applying 20 clusters
    applying 25 clusters
    applying 30 clusters
    applying 35 clusters



```python
# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.

plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE vs. K');
```


![png](output_80_0.png)



```python
scores
```




    [68656673.77467142,
     63310659.966458,
     60575304.883048564,
     58127808.65660498,
     56196627.50945433,
     54419207.460554905,
     52622383.92164255]




```python
# this takes 2-3 hours to run in mac pro

centers2 = range(40, 60, 5)
scores2 = fit_mods(one_hot_data_pca, centers2)
```

    applying 40 clusters
    applying 45 clusters
    applying 50 clusters
    applying 55 clusters



```python
# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
scores2

```




    [51523235.04840434, 50750805.60020616, 49738061.95247968, 49021509.485105716]




```python
score_total = scores + scores2
range_total = range(5, 60, 5)
plt.plot(range_total, score_total, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE vs. K');
```


![png](output_84_0.png)


### Discussion 3.1: Apply Clustering to General Population

(Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)

A:

As we can see from the above figure, the score decrease along with the clusters, but when clusters equal to , it goes slowly, so I took the "elbow" point which is 

### Step 3.2: Apply All Steps to the Customer Data

Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.

- Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
- Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
- Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.


```python
# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', delimiter=';')
```


```python
# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.


```

### Step 3.3: Compare Customer Data to Demographics Data

At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.

Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.

Take a look at the following points in this step:

- Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
  - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
- Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
- Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?


```python
# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.


```


```python
# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?


```


```python
# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?


```

### Discussion 3.3: Compare Customer Data to Demographics Data

(Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)

> Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.
