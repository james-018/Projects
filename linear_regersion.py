"""
Linear Regression Model for House Price Prediction

Dataset: California Housing Prices (from Kaggle)
URL: https://www.kaggle.com/datasets/camnugent/california-housing-prices

Features used:
- total_rooms: Total number of rooms in the house
- housing_median_age: Median age of houses in the block
- median_income: Median income of households in the block
- ocean_proximity: Location category (categorical)
- latitude & longitude: Geographic coordinates (to capture location effects)
- Derived: avg_rooms_per_household = total_rooms / households
- Target: median_house_value

This script:
1. Downloads the dataset from Kaggle (requires Kaggle API setup)
2. Preprocesses: handles missing values, encodes categorical, scales features
3. Splits data
4. Trains Linear Regression
5. Evaluates with RMSE, MAE, R²
6. Makes sample predictions
"""

# Step 1: Install required packages (run once)
# !pip install pandas numpy scikit-learn kaggle matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ______________________________________
# Step 1: Download Dataset from Kaggle
# ______________________________________

def download_california_housing():
    dataset = "camnugent/california-housing-prices"
    if not os.path.exists("housing.csv"):
        print("Downloading dataset from Kaggle...")
        os.system(f"kaggle datasets download -d {dataset} -f housing.csv")
        os.system("unzip housing.csv.zip")
        print("Download complete: housing.csv")
    else:
        print("Dataset already exists: housing.csv")

# Uncomment the line below to download (first time only)
# download_california_housing()

# ______________________________________
# Step 2: Load and Explore Data
# ______________________________________
df = pd.read_csv("housing.csv")
print("Dataset loaded. Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# ______________________________________
# Step 3: Feature Engineering
# ______________________________________
# Handle missing values in total_bedrooms
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Create useful features
df['avg_rooms_per_household'] = df['total_rooms'] / df['households']
df['avg_bedrooms_per_household'] = df['total_bedrooms'] / df['households']

# Use log transformation for skewed target (optional, but improves linearity)
df['median_house_value_log'] = np.log1p(df['median_house_value'])

# Select features
feature_cols = [
    'total_rooms', 'housing_median_age', 'median_income',
    'ocean_proximity', 'latitude', 'longitude',
    'avg_rooms_per_household'
]

X = df[feature_cols]
y = df['median_house_value']  # Use original or df['median_house_value_log'] for log

print("\nSelected Features:")
print(X.head())

# ______________________________________
# Step 4: Preprocessing Pipeline
# ______________________________________
categorical_cols = ['ocean_proximity']
numerical_cols = [col for col in feature_cols if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# ______________________________________
# Step 5: Split Data
# ______________________________________
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ______________________________________
# Step 6: Build and Train Model
# ______________________________________
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model_pipeline.fit(X_train, y_train)
print("\nModel trained successfully!")

# ______________________________________
# Step 7: Predictions and Evaluation
# ______________________________________
y_pred = model_pipeline.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Error (MAE):     ${mae:,.2f}")
print(f"R² Score:                      {r2:.4f}")
print("="*50)

# ______________________________________
# Step 8: Visualization
# ______________________________________
plt.figure(figsize=(12, 5))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted House Prices")

# Residuals
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, bins=50)
plt.xlabel("Residuals ($)")
plt.title("Distribution of Residuals")

plt.tight_layout()
plt.show()

# ______________________________________
# Step 9: Feature Importance (Coefficients)
# ______________________________________
# Get feature names after one-hot encoding
ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_features = ohe.get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_features)

# Get coefficients
coef = model_pipeline.named_steps['regressor'].coef_
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(coef_df.head(10))

# Plot coefficients
plt.figure(figsize=(10, 6))
top_features = coef_df.head(10)
sns.barplot(data=top_features, x='Coefficient', y='Feature')
plt.title("Top 10 Feature Coefficients (Linear Regression)")
plt.axvline(x=0, color='black', linestyle='--')
plt.show()

# ______________________________________
# Step 10: Make a Single Prediction (Example)
# ______________________________________
print("\n" + "="*50)
print("SAMPLE PREDICTION")
print("="*50)

sample_house = pd.DataFrame([{
    'total_rooms': 2000,
    'housing_median_age': 30,
    'median_income': 5.5,
    'ocean_proximity': 'INLAND',
    'latitude': 34.05,
    'longitude': -118.25,
    'avg_rooms_per_household': 6.0
}])

predicted_price = model_pipeline.predict(sample_house)[0]
print(f"Predicted House Price: ${predicted_price:,.2f}")
