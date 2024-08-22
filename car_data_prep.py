import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

def prepare_data(df):
    
    # Convert categorical values
    df['Km'] = df['Km'].replace(',', '')
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce').astype('Int64')
    
    df['capacity_Engine'] = df['capacity_Engine'].replace(',', '')
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce').astype('Int64')
    
    df["manufactor"] = df["manufactor"].replace('Lexsus', 'לקסוס')
    df["Gear"] = df["Gear"].str.replace('אוטומטית','אוטומט')
    df["Engine_type"] = df["Engine_type"].str.replace('היברידי','היבריד')
    
    # Handle missing values
    df['Age'] = 2024 - df['Year']
    df['Km_per_year'] = df['Km'] / df['Age']
    Km_per_year = int(df['Km_per_year'].mean())
    
    df['Km'] = df['Km'].fillna(Km_per_year * df['Age'])
    df['Year'] = df['Year'].fillna((df['Km'] / Km_per_year).astype('Int64'))
       
    numeric_columns = ['Year', 'Hand', 'capacity_Engine', 'Km']
    categorical_columns = ['model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Color', 'Test']    
    
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
    
    def Km_adjust(value):
        if value < 1000:
            return value * 1000
        else:
            return value
    
    df["Km"] = df["Km"].apply(Km_adjust)  
    
    df.loc[df["capacity_Engine"] > 3500, "capacity_Engine"] = 3500
    df.loc[df["capacity_Engine"] < 900, "capacity_Engine"] = 900
    
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    for col in ['Price', 'Km', 'capacity_Engine']:
        df = remove_outliers(df, col)
    
    return df