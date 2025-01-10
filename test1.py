#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:57:23 2025

@author: KalenJosifovski
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# %%

# Load the data from the csv files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# %%

# Convert object columns to category dtype
categorical_cols = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'SaleType', 'SaleCondition'
]

for col in categorical_cols:
    train_data[col] = train_data[col].astype('category')
    test_data[col] = test_data[col].astype('category')

# %%

# Prepare the data splits
x = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the model
model_LGB = lgb.LGBMRegressor(objective='regression', n_estimators=100, learning_rate=0.1, random_state=42)

model_LGB.fit(x_train, y_train, eval_set=[(x_val, y_val)])

#%%

y_val_pred = model_LGB.predict(x_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE: {rmse}")


# Calculate MAE
mae = mean_absolute_error(y_val, y_val_pred)
print(f"MAE: {mae}")

# Calculate MAPE
mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
print(f"MAPE: {mape:.2f}%")

# Calculate explained variance
evs = explained_variance_score(y_val, y_val_pred)
print(f"Explained Variance Score: {evs}")

#%%

# Calculate R² with baseline forecast of zero
r2 = r2_score(y_val, y_val_pred)
print(f"R²: {r2}")

#%%

# Calculate R² with baseline forecast of median of training y values

median_baseline = np.median(y_train)

ss_residual_model = np.sum((y_val - y_val_pred)**2)

ss_total_median = np.sum((y_val - median_baseline)**2)

r2_median_baseline = 1 - (ss_residual_model / ss_total_median)

print(f"R² Against Median Baseline: {r2_median_baseline}")

#%%

# Predict house prices for the unseen test set and save to csv file

results = pd.DataFrame({'Id': test_data['Id']})

results['SalePrice'] = model_LGB.predict(test_data)

results.to_csv('Kalen_Josifovski_submission_results1.csv', index=False, header=True)

# Submitted on 10/01/25. Score = 0.14289, Ranking = 2390/28209, Top 8.47% of the leaderboard

#%%

lgb.plot_importance(model_LGB, max_num_features=20, importance_type='gain')
plt.show()

