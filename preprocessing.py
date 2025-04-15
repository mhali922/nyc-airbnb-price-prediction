from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def load_data():
    # Load dataset
    df = pd.read_csv('data/AB_NYC_2019.csv')  # Adjust path if necessary
    return df

def preprocess_data(df):
    # Example of preprocessing
    df.fillna({'price': df['price'].median()}, inplace=True)  # Fill missing prices with median
    
    # Feature engineering example
    df['room_type'] = df['room_type'].map({'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2})
    df['neighbourhood'] = df['neighbourhood'].astype('category').cat.codes
    
    # Split data into features and target variable
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def create_pipeline():
    numerical_features = ['latitude', 'longitude', 'minimum_nights', 'reviews_per_month']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    return model_pipeline

