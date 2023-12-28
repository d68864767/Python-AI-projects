```python
# Importing necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model():
    """
    This function creates a Sequential model with two Dense layers.
    The first Dense layer has 32 units and uses the relu activation function.
    The second Dense layer has 1 unit and uses the sigmoid activation function.
    The model uses the binary_crossentropy loss function and the adam optimizer.
    """
    # Create a Sequential model
    model = Sequential()

    # Add the first Dense layer
    model.add(Dense(32, activation='relu'))

    # Add the second Dense layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
```
