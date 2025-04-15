import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data():
    # Load the dataset
    df = pd.read_csv('data/airbnb_nyc.csv')
    return df

def preprocess_data(df):
    # Handle missing values, feature encoding, etc.
    df.fillna({'price': df['price'].median()}, inplace=True)  # Example of filling missing price

    # Example: Feature Engineering
    df['room_type'] = df['room_type'].map({'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2})
    df['neighbourhood'] = df['neighbourhood'].astype('category').cat.codes

    # Split data into features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


