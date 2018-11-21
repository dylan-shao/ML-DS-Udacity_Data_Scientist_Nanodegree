

```
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
# Using TensorFlow 1.0.0; use tf.python_io in later versions
tf.python_io.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
# One-hot encoding the output
y = np_utils.to_categorical(y)

print(X.shape)

# Building the model
xor = Sequential()

# Add required layers
xor.add(Dense(8, input_dim=X.shape[1]))
xor.add(Activation('tanh'))

xor.add(Dense(2))
# Add a sigmoid activation layer
xor.add(Activation('sigmoid'))

# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
xor.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])

# Uncomment this line to print the model architecture
xor.summary()

# Fitting the model
history = xor.fit(X, y, epochs=1200, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))
```

    (4, 2)
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_5 (Dense)              (None, 8)                 24        
    _________________________________________________________________
    activation_5 (Activation)    (None, 8)                 0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 2)                 18        
    _________________________________________________________________
    activation_6 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 42
    Trainable params: 42
    Non-trainable params: 0
    _________________________________________________________________
    4/4 [==============================] - 0s 7ms/step
    
    Accuracy:  1.0
    
    Predictions:
    [[ 0.95283729  0.05251895]
     [ 0.11633323  0.78340721]
     [ 0.07509472  0.87407708]
     [ 0.89450121  0.20896456]]

