#!/usr/bin/env python
# coding: utf-8

# # Project - User Car Prices - Predictor
# 
# ## Team Background
# 
# - Project Grp 08
# 
# ### Team Members
# 
# - Team member 1
#     - Name: Richard Anton
#     - Email: [rna63@drexel.edu](mailto:rna63@drexel.edu)
#     

import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import streamlit as st


def load_model(model_path):
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model

# TODO: put in final model path
model_path = 'model_xgboost_s_2023_03_13_07_53_37.pkl'
model = load_model(model_path)

# TODO: use full dataset.
dataset_path = "craigslist_sampled_cleaned_2023_03_05_19_07_36.csv"
target_col = 'price'

df = pd.read_csv(dataset_path)

# show a sample for sanity check
st.write(df.head())

# split into input data and output values
X_all = df.drop(columns=[target_col])
y_all = df[target_col]

st.write("X.shape", X_all.shape)
#print("y.shape", y_all.shape)

# Convert categorical columns to Pandas category type
cats = X_all.select_dtypes(exclude=np.number).columns.tolist()
for col in cats:
    X_all[col] = X_all[col].astype('category')

# It's important that the random_state matches the other notebook.
_, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=193)
# just keep test data

print("X_test.shape", X_test.shape)

numeric_cols = ['year', 'odometer']
cat_cols = ['make', 'model', 'condition', 'cylinders', 'fuel', 'title_status',
            'transmission', 'drive', 'size', 'type', 'paint_color', 'state']

preprocessor_path = 'model_preprocessor_s_2023_03_13_07_28_32.pkl'
preprocess = load_model(preprocessor_path)


def xgboost_predict(X_test):
    X_pre_test = preprocess.transform(X_test)

    # Convert to the format XGBoost lib expects.
    dtest_reg = xgb.DMatrix(X_pre_test)
    predict_test = model.predict(dtest_reg)

    return predict_test


predict_test = xgboost_predict(X_test)

# NOTE: this is just a sanity check. Should not have test values for real life.
st.write('RMSE of test data: ', mean_squared_error(y_test, predict_test) ** (0.5))

r2 = r2_score(y_test, predict_test)
st.write('R2 Score of test data:', r2)
