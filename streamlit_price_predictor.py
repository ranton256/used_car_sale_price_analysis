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
# Run this with python -mstreamlit run streamlit_price_predictor.py
# after installing streamlit package with pip

import joblib
import pandas
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import streamlit as st

preprocessor_path = 'model_preprocessor_f_2023_03_13_14_22_04.pkl'
model_path = "model_xgboost_2023_03_13_16_35_43.pkl"

sampled = False
if sampled:
    dataset_path = "craigslist_sampled_cleaned_2023_03_05_19_07_36.csv"
else:  # Full dataset
    dataset_path = "craigslist_full_cleaned_2023_03_12_10_45_22.csv"

target_col = 'price'

# numeric_cols = ['year', 'odometer']
# cat_cols = ['make', 'model', 'condition', 'cylinders', 'fuel', 'title_status',
#             'transmission', 'drive', 'size', 'type', 'paint_color', 'state']


def load_model(model_path):
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model



def xgboost_predict(X_test):
    X_pre_test = preprocess.transform(X_test)

    # Convert to the format XGBoost lib expects.
    dtest_reg = xgb.DMatrix(X_pre_test)
    predict_test = model.predict(dtest_reg)

    return predict_test


def load_test_data(ds_path):
    df = pd.read_csv(ds_path)

    # show a sample for sanity check
    # st.write(df.head())

    # split into input data and output values
    X_all = df.drop(columns=[target_col])
    y_all = df[target_col]

    st.write("X.shape", X_all.shape)

    # Convert categorical columns to Pandas category type
    cats = X_all.select_dtypes(exclude=np.number).columns.tolist()
    for col in cats:
        X_all[col] = X_all[col].astype('category')

    return X_all, y_all


def run_sanity_check():
    st.write("Running sanity check on model")

    X_all, y_all = load_test_data(dataset_path)

    # It's important that the random_state matches the other notebook.
    # just keep test data
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=193)
    st.write("X_test.shape", X_test.shape)

    predict_test = xgboost_predict(X_test)

    st.write('RMSE of test data: ', mean_squared_error(y_test, predict_test) ** (0.5))
    r2 = r2_score(y_test, predict_test)
    st.write('R2 Score of test data:', r2)


INPUT_COLUMNS= [
    "year",
    "make",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "drive",
    "size",
    "type",
    "paint_color",
    "state"
    ]


def make_input_df():
    d = {cn: st.session_state.get(cn) for cn in INPUT_COLUMNS}
    st.write(d)
    df = pandas.DataFrame(data=d, index=[0])
    return df


if __name__ == "__main__":
    st.write("Used car price predictor")

    # TODO: figure out how to cache this instead
    # of running every time
    # see  @st.cache_data
    preprocess = load_model(preprocessor_path)
    model = load_model(model_path)
    st.write("Model loaded")
    run_sanity_check()




    st.text_input('Make', key='make')
    st.text_input('Model', key='model')
    # TODO: drop downs.
    st.text_input('Condition', key='condition')
    st.text_input('Cylinders', key='cylinders')
    st.text_input('Fuel', key='fuel')
    st.text_input('Title Status', key='title_status')
    st.text_input('Transmission', key='transmission')
    st.text_input('Drive', key='drive')
    st.text_input('Size', key='size')
    st.text_input('Type', key='type')
    st.text_input('Paint Color', key='paint_color')
    st.text_input('State', key='state')

    # TODO; defaults
    # TODO: integer
    st.number_input('Year', key='year')
    st.number_input('Mileage', key='odometer')

    # show our data
    #st.write("You entered:")
    #values = [f"{k}={v}" for k, v in st.session_state.items()]
    #st.write(values)

    df = make_input_df()
    st.write(df)