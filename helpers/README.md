
## What is this
This is the helpers of numpy when I learn the Udacity Data Scientist Nanodegree. Some of them have examples, some of them are just links to other people's answers. 
Hope this will help people who is new to python and numpy.

-------


```python
#helper
def line():
    print('\n-----------------\n')
```

### [Random seed](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.seed.html)
Seed the generator.


```python
import numpy as np

print(np.random.rand(4))
print(np.random.rand(4))
```

    [0.37511719 0.86003838 0.42883142 0.47422804]
    [0.30941275 0.86997749 0.22477763 0.99611517]



```python
import numpy as np

np.random.seed(1)
print(np.random.rand(4))

np.random.seed(1)
print(np.random.rand(4))
```

    [4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01]
    [4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01]


------

### [Difference between numpy.array shape (R, 1) and (R,)](https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r)


```python
import util

a = np.arange(12)

util.shapePrintHelper(a)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11]
    ----------
    shape: (12,)



```python
# reshape it to (6, 2)
b = a.reshape(6,2)

util.shapePrintHelper(b)
```

    [[ 0  1]
     [ 2  3]
     [ 4  5]
     [ 6  7]
     [ 8  9]
     [10 11]]
    ----------
    shape: (6, 2)



```python
# reshape it to (1, 12)
c = b.reshape(1, 12)
util.shapePrintHelper(c)
```

    [[ 0  1  2  3  4  5  6  7  8  9 10 11]]
    ----------
    shape: (1, 12)



```python
# will have error if reshape to a shape (x,y) that x*y !== number of element

d = c.reshape(3, 3)
util.shapePrintHelper(d)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-7-2698f9c73634> in <module>()
          1 # will have error if reshape to a shape (x,y) that x*y !== number of element
          2 
    ----> 3 d = c.reshape(3, 3)
          4 util.shapePrintHelper(d)


    ValueError: cannot reshape array of size 12 into shape (3,3)


-----------


```python
# np.float_ takes list/string/number as params, np.float takes string/number
a = [1,2,3]
print(np.float_(a))
line()

# error
print(np.float(a))
```

    [1. 2. 3.]
    
    -----------------
    



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-21-8d59151d05a2> in <module>()
          5 
          6 # error
    ----> 7 print(np.float(a))
    

    TypeError: float() argument must be a string or a number, not 'list'

