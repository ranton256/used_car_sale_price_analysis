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


@st.cache_resource
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


@st.cache_data
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

    return predict_test


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

CONDITIONS = ['good', 'excellent', 'fair', 'like new', 'new', 'salvage']
CYLINDERS = ['8 cylinders', '6 cylinders', '4 cylinders', '5 cylinders',
             'other', '3 cylinders', '10 cylinders', '12 cylinders']
FUELS = ['gas', 'other', 'diesel', 'hybrid', 'electric']
TITLE_STATUSES = ['clean', 'rebuilt', 'lien', 'salvage', 'missing', 'parts only']
TRANSMISSIONS = ['other', 'automatic', 'manual']
DRIVES = ['rwd', '4wd', 'fwd']
SIZES = ['full-size', 'mid-size', 'compact', 'sub-compact']
TYPES = ['pickup', 'truck', 'other', 'coupe', 'SUV', 'hatchback',
         'mini-van', 'sedan', 'offroad']
PAINT_COLORS = ['white', 'blue', 'red', 'black', 'silver', 'grey', 'brown',
                'yellow', 'orange']


# TODO: states

def make_input_df():
    d = {cn: st.session_state.get(cn) for cn in INPUT_COLUMNS}
    st.write(d)
    df = pandas.DataFrame(data=d, index=[0])
    return df


def setup_controls():

    col1, col2 = st.columns(2)

    with col1:
        st.text_input('Make', key='make')  # TODO: put in select for make.
        st.text_input('Model', key='model')  # free-form because unique values too high.
        st.selectbox('Condition', CONDITIONS, key='condition')
        st.selectbox('Cylinders', CYLINDERS, key='cylinders')
        st.selectbox('Fuel', FUELS, key='fuel')
        st.selectbox('Title Status', TITLE_STATUSES, key='title_status')
        st.selectbox('Transmission', TRANSMISSIONS, key='transmission')

    with col2:
        st.selectbox('Drive', DRIVES, key='drive')
        st.selectbox('Size', SIZES, key='size')
        st.selectbox('Type', TYPES, key='type')
        st.selectbox('Paint Color', PAINT_COLORS, key='paint_color')
        st.text_input('State', key='state')  # TODO: states
        st.number_input('Year', key='year', min_value=1900, max_value=2025, format="%d")
        st.number_input('Mileage', key='odometer', value=0, min_value=0, max_value=1000000, format="%d")


if __name__ == "__main__":
    st.title("Used car price predictor")

    preprocess = load_model(preprocessor_path)
    model = load_model(model_path)


    st.write("Model loaded")

    st.header("Sanity Check")
    run_sanity_check()

    st.header("Enter vehicle data")
    setup_controls()

    df = make_input_df()
    st.write(df)

    st.header("Predicted price")
    predicted = xgboost_predict(df)
    # TODO: predicted as scalar
    st.write(f"${predicted}")

