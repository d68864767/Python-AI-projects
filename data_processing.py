```python
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    """
    This function loads the data from a csv file.
    """
    # Load the data
    data = pd.read_csv('data.csv')

    return data

def process_data(data):
    """
    This function processes the data by splitting it into features and labels.
    """
    # Split the data into features and labels
    X = data.drop('target', axis=1)
    y = data['target']

    return {'X': X, 'y': y}
```
