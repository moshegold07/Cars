#!/usr/bin/env python
# coding: utf-8

# ##### מגישים:
# ##### משה גולדזנד ת.ז 312486046
# ##### מנחם פרל ת.ז 318836962
# ##### https://github.com/moshegold07/Cars

# ##### Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error


# ##### Define the function to prepare data
# prepare_data(df)
# The prepare_data function is designed to clean and preprocess a DataFrame (df) to prepare it for machine learning models. It involves several key steps to ensure the data is in a suitable format for analysis. Here’s a detailed breakdown of each step:
# 
# Remove Outliers in Price:
# This step filters out entries with prices that fall outside a reasonable range. Specifically, it excludes prices that are below 4000 or above the 99th percentile of the price distribution. This helps in eliminating extreme values that could skew the model’s predictions. Additionally, any rows with missing price values are removed.
# Convert Categorical Values to Numeric:
# 
# The function ensures that numeric values stored as strings are converted into proper numeric formats. For example, it removes commas from values representing kilometers and engine capacity, then converts these values to floating-point numbers. This conversion is crucial for numerical computations in subsequent steps.
# 
# Handle Missing Values:
# Missing values in both numeric and categorical columns are addressed to avoid errors during modeling. For numeric columns, missing values are replaced with the median value of each column. For categorical columns, the most frequent value is used to fill in missing entries. This ensures that the dataset is complete and ready for analysis.
# Encode Categorical Variables:
# 
# Categorical variables, which are typically non-numeric, are converted into numeric codes. This transformation allows these variables to be used effectively in machine learning algorithms that require numerical input.
# 
# Feature Engineering:
# New features are created to enhance the model's ability to make predictions. This includes calculating the age of the car based on the year of manufacture, the average kilometers driven per year, and the price per kilometer. These additional features provide more context and potentially improve the predictive performance of the model.
# 
# Prepare Feature and Target Variables:
# The function separates the features (predictor variables) from the target variable (the value to be predicted). This separation is essential for training machine learning models, as it allows the model to learn from the features and make predictions based on the target variable.
# 
# Remove Additional Outliers:
# Further cleaning is performed by removing rows with extreme values, identified through z-scores. Z-scores are used to measure how far each data point is from the mean, and any data points with z-scores above a certain threshold (3 in this case) are considered outliers and removed.
# 
# Return Processed Data:
# Finally, the function returns the cleaned and processed feature matrix and target vector. This prepared data can now be used for training and evaluating machine learning models.

# In[2]:


def prepare_data(df):
    # הסרת ערכים חריגים במחיר
    df = df[(df['Price'] >= 4000) & (df['Price'] <= df['Price'].quantile(0.99))]
    df = df.dropna(subset=['Price'])

    # Remove outliers in price    
    df['Km'] = df['Km'].str.replace(',', '').astype(float, errors='ignore')
    df['capacity_Engine'] = df['capacity_Engine'].str.replace(',', '').astype(float, errors='ignore')

    # Convert categorical values to numeric
    numeric_columns = ['Year', 'Hand', 'capacity_Engine', 'Km']
    categorical_columns = ['model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Color', 'Test']

    # Handle missing values
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    # Encode categorical variables
    for feature in categorical_columns:
        df[feature] = pd.Categorical(df[feature]).codes

    # Feature engineering
    df['Age'] = 2024 - df['Year']
    df['Km_per_year'] = df['Km'] / df['Age']
    df['Price_per_km'] = df['Price'] / df['Km']

    features = ['Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Km',
                'Prev_ownership', 'Curr_ownership', 'Age', 'Km_per_year', 'Color', 'Test']

    X = df[features]
    y = df['Price']

    # Remove additional outliers
    z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
    X = X[(z_scores < 3).all(axis=1)]
    y = y[X.index]

    return X, y


# #### Reading the Data
# ###### This section reads data from a CSV file named dataset.csv into a Pandas DataFrame (df). The DataFrame will be used for data processing and model training.

# In[3]:


df = pd.read_csv('dataset.csv')


# #### Preparing the Data
# ###### Here, the prepare_data function is called to preprocess the DataFrame (df). This function cleans and transforms the data, splitting it into feature variables (X) and the target variable (y). The cleaned data is then ready for modeling.

# In[4]:


X, y = prepare_data(df)


# #### Splitting the Data
# ###### This line splits the dataset into training and testing subsets. 80% of the data is used for training the model (X_train and y_train), and 20% is reserved for evaluating the model (X_test and y_test). The random_state parameter ensures reproducibility of the split.

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Creating the Pipeline
# ###### A pipeline is constructed to streamline the modeling process. This pipeline includes:
# Scaler: Standardizes features by removing the mean and scaling to unit variance.
# Polynomial Features: Generates polynomial and interaction features of degree 2.
# Feature Selection: Selects important features using ElasticNet as a base model.
# Model: Applies ElasticNet regression for the final prediction.

# In[6]:


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('feature_selection', SelectFromModel(ElasticNet(random_state=42))),
    ('model', ElasticNet(random_state=42))
])


# #### Defining Hyperparameters for Search
# ###### This dictionary defines the parameter grid for hyperparameter tuning. It specifies:
# Polynomial Degree: Degrees to consider for polynomial features.
# Feature Selection: Number of features to select.
# ElasticNet Parameters: Values for alpha (regularization strength) and l1_ratio (balance between L1 and L2 regularization).

# In[7]:


param_grid = {
    'poly__degree': [1, 2, 3],
    'feature_selection__max_features': list(range(1, X_train.shape[1] + 1)),
    'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
}


# #### Performing Randomized Search with Cross-Validation
# ###### This step performs a Randomized Search with Cross-Validation to find the best hyperparameters for the pipeline. It evaluates different combinations of parameters using 10-fold cross-validation and measures performance with negative mean squared error.

# In[8]:


random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid,
                                   n_iter=100, cv=10, scoring='neg_mean_squared_error',
                                   random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)


# #### Printing Best Parameters and Cross-Validation Score
# ###### The best parameters found by the Randomized Search are printed, along with the best cross-validation score, converted to root mean squared error (RMSE) for easier interpretation.

# In[9]:


print("Best parameters:", random_search.best_params_)
print("Best cross-validation score: ", np.sqrt(-random_search.best_score_))


# #### Evaluating the Final Model
# ###### The final model, selected by Randomized Search, is used to predict the test set outcomes. The RMSE of these predictions is computed and printed to assess the model's performance on unseen data.

# In[10]:


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nTest RMSE:", rmse)


# #### Identifying Important Features
# ###### This section extracts and displays the most important features based on the trained model. It identifies the features selected by the model and their associated importance scores, along with their positive or negative impact on the target variable. The top 5 features are printed.

# In[11]:


feature_selector = best_model.named_steps['feature_selection']
all_feature_names = best_model.named_steps['poly'].get_feature_names_out(X.columns)
selected_features = all_feature_names[feature_selector.get_support()]

feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': abs(best_model.named_steps['model'].coef_),
    'sign': np.sign(best_model.named_steps['model'].coef_)
})

feature_importance['sign'] = feature_importance['sign'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(feature_importance.head(5))


# #### Cross-Validation of the Final Model
# ###### Finally, the model's performance is evaluated using cross-validation on the entire dataset. The RMSE scores from cross-validation are computed and displayed, including the mean and standard deviation, to assess the model's stability and generalization capability.

# In[12]:


cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
print("\nCross-validation RMSE scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Standard deviation of RMSE:", rmse_scores.std())

