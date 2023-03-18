# README - Used Car Price Regression

## Team

- Project Grp 08

### Team Members

- Team member 1
  - Name: Richard Anton
  - Email: [rna63@drexel.edu](mailto:rna63@drexel.edu)

## Dataset

Craigslist: <https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data>

This dataset was created by the provider scraping data from Craigslist car listings.
The columns include price, condition, manufacturer, and latitude/longitude plus 18 other categories.

## Geospatial data

TODO: remove this section if we don't use the data.

The US State geo boundary data was obtained from:

<https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html>

## Notebooks

### Data Preparation Notebook - rna63_project_part2.ipynb

Exploratory data analysis, cleaning, and preparation is largely contained
in the notebook rna63_project_part2.ipynb.
This file contains code to save the preprocessed, cleaned data to a CSV file for use in regression models.

### Model Training Notebook - used_car_price_regression.ipynb

This notebook contains the code for training all of the regression models
on the preprocessed CSV data output from the notebook describe above.
It also saves the final XGBoost model file after training.

### Inference Notebook - used_car_price_predictor.ipynb

This notebook loads the model trained from the regression notebook
and uses it along with user inputs to predict the listing price for
a used vehicle. This is mainly for troubleshooting locally.

## Streamlit notebook - streamlit_price_predictor.py 

There is a streamlit notebook which loads a saved model and allows
the user to input vehicle data then predict the listing price using
the trained model.

 TODO: Put in link to Streamlit app online.

## Dependencies

- The main dependencies for the notebooks and streamlist app include:

- joblib (used to save and load models)
- numpy
- pandas
- scikit-learn
- xgboost
- category_encoders
- matplotlib
- streamlit
- watchdog (This is optional for streamlit)
- s3fs (for Streamlit to load model files)


There is a 'requirements.txt' files that can be used with pip to install
all the dependencies into a virtualenv 


## Setup

1. [ ] Install Python package dependencies into your environment.
   1. You can create a new virtualenv with 'python -mvenv .venv' or use Conda/Miniconda
   2. Run 'pip install -r requirements.txt' from the unzipped project directory.
2. [ ] Download dataset from <https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data>
  - then unzip the archive.zip file.
3. [ ] Create the datasets/craigslist directory 
   1. In the same directory as the Jupyter notebook files
   2. 'mkdir -p datasets/craigslist'
4. [ ] Move the vehicles.csv file from the zip file to the datasets/craigslist directory
   1. 'mv vehicles.csv datasets/craigslist'

## Data Preparation

Use the 'rna63_project_part2.ipynb' Jupyter notebook to load the dataset, plot various graphs and 
statistics used for the exploratory data analysis, and output the cleaned data to a new CSV file.

Run all the cells in the notebook.

Look at the output of the last cell in the notebook to find the name of the output CSV.
It will say something like 'Saving cleaned dataframe to your_real_file_name.csv'.

## Model Training

Use the 'used_car_price_regression.ipynb' Jupyter notebook to train and validate the regression models
used in the project and generate the feature importance plot.

Make sure the 'sampled' variable is set to False.

Find the cell where 'dataset_path' is assigned and update it to the CSV filename output
by the 'rna63_project_part2.ipynb'.

    if sampled:
        dataset_path = "craigslist_sampled_cleaned_2023_03_05_19_07_36.csv"
    else:  # Full dataset
        dataset_path = "craigslist_full_cleaned_2023_03_12_10_45_22.csv"
    

Run all the cells in the notebook.

Observe the generated statistics, and the training graph for the final XGBoost model.

In order to use the trained model in the inference notebook or the streamlit app,
find the filenames printed by the save_model() function for both the preprocesor and the
actual XGBoost model.  There should be cell outputs similar containing text similar to this:

    Saving model to model_preprocessor_f_2023_03_13_14_22_04.pkl
    ...
    Saving model to model_xgboost_2023_03_13_16_35_43.pkl

## Price Prediction Using Trained Model.

The Jupyter notebook 'used_car_price_predictor.ipynb' contains code which loads the saved model,
recreates the same train/test split, and then runs the test set through the model and outputs
the same statistics as the training notebook.

TODO: how to update model filenames.


This is just a demonstration of how to load and use the saved model, and a sanity check that
it produces the same results as the training notebook on the same data.

## Interactive Price Prediction Using Streamlit App

The Streamlit app contained in 'streamlit_price_predictor.py' allows entering the input 
values in a web interface and generating a predicted price using the saved model.

TODO: update to use model filenames.

The app can be run locally with:

    python -mstreamlit run streamlit_price_predictor.py

When run locally, if no environment variables containing S3 

TODO: fix streamlit app to use local files when nothing in env.

The app is also deployed to a Streamlit community cloud app at <TODO: put in real URL>
The deployed cloud app uses model files written to an S3 bucket since they are too large
to check into the GitHub repository.








