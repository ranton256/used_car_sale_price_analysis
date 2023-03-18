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

import s3fs
import os

preprocessor_path = 'model_preprocessor_f_2023_03_13_14_22_04.pkl'
model_path = "model_xgboost_2023_03_13_16_35_43.pkl"
dataset_path = "craigslist_full_cleaned_2023_03_12_10_45_22.csv"

target_col = 'price'

bucket_name = 'used-car-price-predictor'


@st.cache_resource
def load_model(model_path, _fs=None):
    if _fs is None:
        print(f"Loading model from {model_path}")
        loaded = joblib.load(model_path)
    else:
        s3_path = bucket_name + '/' + model_path
        with _fs.open(s3_path) as f:
            loaded = joblib.load(f)

    return loaded


def xgboost_predict(X_test):
    X_pre_test = preprocess.transform(X_test)

    # Convert to the format XGBoost lib expects.
    dtest_reg = xgb.DMatrix(X_pre_test)
    predict_test = model.predict(dtest_reg)

    return predict_test

@st.cache_resource
def load_test_data(ds_path, _fs=None):

    if _fs is None:
        df = pd.read_csv(ds_path)
    else:
        st.write("Loading test data from S3")
        #AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        #AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        # AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

        # Pandas supports using s3fs to read from S3.
        df = pd.read_csv(
            f"s3://{bucket_name}/{ds_path}",
            # storage_options={
            #     "key": AWS_ACCESS_KEY_ID,
            #     "secret": AWS_SECRET_ACCESS_KEY,
            #     #"token": AWS_SESSION_TOKEN,
            # },
        )

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
def run_sanity_check(_fs):
    st.write("Running sanity check on model")

    X_all, y_all = load_test_data(dataset_path, _fs=_fs)

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


def get_states():
    columns = ["state","abbrev","code"]
    state_data = [
        ["Alabama","Ala.","AL"],
        ["Alaska","Alaska","AK"],
        ["Arizona","Ariz.","AZ"],
        ["Arkansas","Ark.","AR"],
        ["California","Calif.","CA"],
        ["Colorado","Colo.","CO"],
        ["Connecticut","Conn.","CT"],
        ["Delaware","Del.","DE"],
        ["District of Columbia","D.C.","DC"],
        ["Florida","Fla.","FL"],
        ["Georgia","Ga.","GA"],
        ["Hawaii","Hawaii","HI"],
        ["Idaho","Idaho","ID"],
        ["Illinois","Ill.","IL"],
        ["Indiana","Ind.","IN"],
        ["Iowa","Iowa","IA"],
        ["Kansas","Kans.","KS"],
        ["Kentucky","Ky.","KY"],
        ["Louisiana","La.","LA"],
        ["Maine","Maine","ME"],
        ["Maryland","Md.","MD"],
        ["Massachusetts","Mass.","MA"],
        ["Michigan","Mich.","MI"],
        ["Minnesota","Minn.","MN"],
        ["Mississippi","Miss.","MS"],
        ["Missouri","Mo.","MO"],
        ["Montana","Mont.","MT"],
        ["Nebraska","Nebr.","NE"],
        ["Nevada","Nev.","NV"],
        ["New Hampshire","N.H.","NH"],
        ["New Jersey","N.J.","NJ"],
        ["New Mexico","N.M.","NM"],
        ["New York","N.Y.","NY"],
        ["North Carolina","N.C.","NC"],
        ["North Dakota","N.D.","ND"],
        ["Ohio","Ohio","OH"],
        ["Oklahoma","Okla.","OK"],
        ["Oregon","Ore.","OR"],
        ["Pennsylvania","Pa.","PA"],
        ["Rhode Island","R.I.","RI"],
        ["South Carolina","S.C.","SC"],
        ["South Dakota","S.D.","SD"],
        ["Tennessee","Tenn.","TN"],
        ["Texas","Tex.","TX"],
        ["Utah","Utah","UT"],
        ["Vermont","Vt.","VT"],
        ["Virginia","Va.","VA"],
        ["Washington","Wash.","WA"],
        ["West Virginia","W.Va.","WV"],
        ["Wisconsin","Wis.","WI"],
        ["Wyoming","Wyo.","WY"]
    ]
    states_df = pd.DataFrame(data=state_data, columns=columns)
    return states_df


def make_input_df():
    d = {cn: st.session_state.get(cn) for cn in INPUT_COLUMNS}
    st.write(d)
    df = pandas.DataFrame(data=d, index=[0])
    return df


def setup_controls():

    states = get_states()
    state_codes = states['code'].values.tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.text_input('Make', key='make')  # TODO: put in select for make.
        st.text_input('Model', key='model')  # free-form because unique values too high.
        st.selectbox('Condition', CONDITIONS, key='condition')
        st.selectbox('Cylinders', CYLINDERS, key='cylinders')
        st.selectbox('Fuel', FUELS, key='fuel')
        st.selectbox('Title Status', TITLE_STATUSES, key='title_status',
                     index=TITLE_STATUSES.index("clean"))
        st.selectbox('Transmission', TRANSMISSIONS, key='transmission',
                     index=TRANSMISSIONS.index("automatic"))

    with col2:
        st.selectbox('Drive', DRIVES, key='drive', index=DRIVES.index('fwd'))
        st.selectbox('Size', SIZES, key='size', index=SIZES.index('mid-size'))
        st.selectbox('Type', TYPES, key='type', index=TYPES.index('sedan'))
        st.selectbox('Paint Color', PAINT_COLORS, key='paint_color')
        st.selectbox('State', state_codes, key='state')
        st.number_input('Year', key='year', min_value=1900, max_value=2025, format="%d", value=2020)
        st.number_input('Mileage', key='odometer', value=100000, min_value=0, max_value=1000000, format="%d")


if __name__ == "__main__":
    st.title("Used Car Price Predictor")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Drexel DSCI 521")
    with col2:
        st.subheader("Richard Anton")
    with col3:
        st.subheader("rna63@drexel.edu")

    if "AWS_ACCESS_KEY_ID" not in os.environ:
        st.error("no access key!")

    # NOTE: just set fs=None to use local files instead of S3.
    fs = s3fs.S3FileSystem(anon=False)

    preprocess = load_model(preprocessor_path, _fs=fs)
    model = load_model(model_path, _fs=fs)
    st.write("Model loaded")

    with st.expander("Sanity Check"):
        run_sanity_check(_fs=fs)

    st.header("Enter vehicle data")
    setup_controls()

    with st.expander("Your Inputs"):
        df = make_input_df()

    st.header("Predicted price")
    results = xgboost_predict(df)
    predicted = results[0]

    st.subheader("${price:.2f}".format(price=predicted))

