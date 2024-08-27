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
from car_data_prep import prepare_data
import pickle

df = pd.read_csv('dataset.csv')
df = prepare_data(df)

# Define the Function to training
def training(df):
    features = ['Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Km',
                'Prev_ownership', 'Curr_ownership']

    X = df[features]
    y = df['Price']

    # Remove additional outliers using Z-scores
    z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
    X = X[(z_scores < 3).all(axis=1)]
    y = y[X.index]
    
    return X, y

X, y = training(df)

#### Splitting the Data
###### This line splits the dataset into training and testing subsets. 80% of the data is used for training the model (X_train and y_train), and 20% is reserved for evaluating the model (X_test and y_test). The random_state parameter ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Age', 'Km_per_year']
categorical_features = ['model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Color', 'Test']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

#### Creating the Pipeline
###### A pipeline is constructed to streamline the modeling process. This pipeline includes:
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('feature_selection', SelectFromModel(ElasticNet(random_state=42))),
    ('model', ElasticNet(random_state=42))
])

#### Defining Hyperparameters for Search
###### This dictionary defines the parameter grid for hyperparameter tuning. It specifies:
param_grid = {
    'poly__degree': [1, 2, 3],
    'feature_selection__max_features': list(range(1, X_train.shape[1] + 1)),
    'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid,
                                   n_iter=100, cv=10, scoring='neg_mean_squared_error',
                                   random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nTest RMSE:", rmse)

# Saving the trained model to a PKL file
pickle.dump(best_model, open('trained_model.pkl', 'wb'))