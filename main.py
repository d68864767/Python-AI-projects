```python
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils import preprocess_data
from models import create_model
from data_processing import load_data, process_data
from tests import run_tests

def main():
    # Load and preprocess the data
    print("Loading and preprocessing data...")
    data = load_data()
    data = preprocess_data(data)

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'], test_size=0.2, random_state=42)

    # Create the model
    print("Creating model...")
    model = create_model()

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model achieved {accuracy*100:.2f}% accuracy on test set.")

    # Run tests
    print("Running tests...")
    run_tests()

if __name__ == "__main__":
    main()
```
