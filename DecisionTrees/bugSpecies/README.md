

```python
import pandas as pd

data = pd.read_csv('data.csv')

brownMo = 0
brownLo = 0

blueMo = 0
blueLo = 0  

greenMo = 0
greenLo = 0  

l17Mo = 0
l17Lo = 0   
l17Mo2 = 0
l17Lo2 = 0   

l20Mo = 0
l20Lo = 0 

moBug = 0
loBug = 0
for index, row in data.iterrows():
    if row[0] == 'Mobug':
        moBug += 1
    else:
        loBug += 1
    #----- add brown -----------
    
    if row[1] == 'Brown':
        if row[0] == 'Mobug':
            brownMo += 1
        else:
            brownLo += 1

    #----- add blue --------------   
    
    if row[1] == 'Blue':  
        if row[0] == 'Mobug':
            blueMo += 1
        else:
            blueLo += 1
    #-----------add green-----------
     
    if row[1] == 'Green':  
        if row[0] == 'Mobug':
            greenMo += 1
        else:
            greenLo += 1

    #-----------add < 17-----------
    
    if row[2] < 17 :  
        if row[0] == 'Mobug':
            l17Mo += 1
        else:
            l17Lo += 1
    else:
        if row[0] == 'Mobug':
            l17Mo2 += 1
        else:
            l17Lo2 += 1
     
            
    #-----------add < 20-----------
     
    if row[2] > 17 and row[2] < 20:  
        if row[0] == 'Mobug':
            l20Mo += 1
        else:
            l20Lo += 1

    
      

import math

def multiEntropy(*args):
  total = sum(args)
  result = 0
  for arg in args:
    p = arg/total
    result -= p*math.log(p,2)
  return result

# print(multiEntropy(brownMo,brownLo))
# print(multiEntropy(blueMo,blueLo))
# print(multiEntropy(greenMo,greenLo))
totalEntropy = multiEntropy(moBug,loBug)
child1Entropy = multiEntropy(l17Mo,l17Lo)
child2Entropy = multiEntropy(l17Mo2,l17Lo2)

# print(multiEntropy(l20Mo,l20Lo))
def calcRes(below, above, total):
    
p1 = (l17Mo + l17Lo)/(moBug+loBug)
p2 = (l17Mo2 + l17Lo2)/(moBug+loBug)
totalEntropy - p1*child1Entropy - p2*child2Entropy
```




    0.11260735516748943


