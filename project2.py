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


# In[2]:


df = pd.read_csv('dataset.csv')


# In[3]:


df.head(2)


# #### Checking for Unique Values
# In this part, we check all the unique values to make sure there are no cases where the same type of data has different names. This helps ensure that the data is consistent and reliable.

# In[4]:


df["manufactor"].unique()


# In[5]:


df["model"].unique()


# In[6]:


df["Gear"].unique()


# In[7]:


df["capacity_Engine"].unique()


# In[8]:


df["Engine_type"].unique()


# In[9]:


df["Price"].unique()


# In[10]:


df['Km'].unique()


# #### prepare_data Function Breakdown
# ###### Define the Function to Prepare Data
# The function prepare_data(df) is defined to process and clean the input DataFrame (df) for machine learning model training. The function takes the raw data as input and outputs the prepared features (X) and the target variable (y).
# 
# Data Cleaning
# The function first removes commas from the Km and capacity_Engine columns and converts them to numeric types. Specific corrections are made to standardize values in the manufactor, Gear, and Engine_type columns.
# 
# Handle Missing Values
# The function calculates the Age of the car based on the Year column. A new feature, Km_per_year, is calculated as Km divided by Age.
# 
# Missing values in the Km column are filled based on the average Km_per_year multiplied by Age. Missing values in the Year column are filled based on the calculated Km divided by the average Km_per_year.
# 
# Numeric columns (Year, Hand, capacity_Engine, Km) are filled with median values. Categorical columns are filled with the most frequent values in the dataset.
# 
# Feature Engineering
# If Km is less than 1000, the value is multiplied by 1000 to ensure consistent scaling. Values of capacity_Engine greater than 3500 are capped at 3500, and those below 900 are raised to 900.
# 
# Outlier Removal
# For the columns Price, Km, and capacity_Engine, the Interquartile Range (IQR) method is used to remove outliers. Further outliers are removed using Z-scores, keeping only data points within three standard deviations of the mean.
# 
# Encoding Categorical Variables
# Categorical variables are encoded using target encoding, where the mean Price for each category is used as the encoding value.
# 
# Feature Scaling
# All numeric features are scaled using StandardScaler to ensure consistent contribution across all features in the model.
# 
# Return Prepared Data
# The function returns the prepared features (X) and the target variable (y).

# In[11]:


# Define the Function to Prepare Data
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
    
    # Encode categorical variables with target encoding
    for feature in categorical_columns:
        df[feature] = df.groupby(feature)['Price'].transform('mean')
    
    features = ['Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 'Km',
                'Prev_ownership', 'Curr_ownership']

    X = df[features]
    y = df['Price']

    # Remove additional outliers using Z-scores
    z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
    X = X[(z_scores < 3).all(axis=1)]
    y = y[X.index]
    
    # Feature scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y


# In[12]:


# #### Reading the Data
# ###### This section reads data from a CSV file named dataset.csv into a Pandas DataFrame (df). The DataFrame will be used for data processing and model training.

# In[13]:


df = pd.read_csv('dataset.csv')


# #### Preparing the Data
# ###### Here, the prepare_data function is called to preprocess the DataFrame (df). This function cleans and transforms the data, splitting it into feature variables (X) and the target variable (y). The cleaned data is then ready for modeling.

# In[14]:


X, y = prepare_data(df)


# #### Splitting the Data
# ###### This line splits the dataset into training and testing subsets. 80% of the data is used for training the model (X_train and y_train), and 20% is reserved for evaluating the model (X_test and y_test). The random_state parameter ensures reproducibility of the split.

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 

# In[16]:


numeric_features = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Age', 'Km_per_year']
categorical_features = ['model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Color', 'Test']


# 

# In[17]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# #### Creating the Pipeline
# ###### A pipeline is constructed to streamline the modeling process. This pipeline includes:
# Scaler: Standardizes features by removing the mean and scaling to unit variance.
# Polynomial Features: Generates polynomial and interaction features of degree 2.
# Feature Selection: Selects important features using ElasticNet as a base model.
# Model: Applies ElasticNet regression for the final prediction.

# In[18]:


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

# In[19]:


param_grid = {
    'poly__degree': [1, 2, 3],
    'feature_selection__max_features': list(range(1, X_train.shape[1] + 1)),
    'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1]
}


# #### Performing Randomized Search with Cross-Validation
# ###### This step performs a Randomized Search with Cross-Validation to find the best hyperparameters for the pipeline. It evaluates different combinations of parameters using 10-fold cross-validation and measures performance with negative mean squared error.

# In[20]:


random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid,
                                   n_iter=100, cv=10, scoring='neg_mean_squared_error',
                                   random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)


# #### Printing Best Parameters and Cross-Validation Score
# ###### The best parameters found by the Randomized Search are printed, along with the best cross-validation score, converted to root mean squared error (RMSE) for easier interpretation.

# In[21]:


print("Best parameters:", random_search.best_params_)
print("Best cross-validation score: ", np.sqrt(-random_search.best_score_))


# #### Evaluating the Final Model
# ###### The final model, selected by Randomized Search, is used to predict the test set outcomes. The RMSE of these predictions is computed and printed to assess the model's performance on unseen data.

# In[22]:


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nTest RMSE:", rmse)


# #### Identifying Important Features
# ###### This section extracts and displays the most important features based on the trained model. It identifies the features selected by the model and their associated importance scores, along with their positive or negative impact on the target variable. The top 5 features are printed.

# In[23]:


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

# In[24]:


cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
print("\nCross-validation RMSE scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Standard deviation of RMSE:", rmse_scores.std())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




