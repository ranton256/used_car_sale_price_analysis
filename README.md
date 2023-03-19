# README - Used Car Price Regression

## Team

- Project Grp 08

### Team Members

- Team member 1
  - Name: Richard Anton
  - Email: [rna63@drexel.edu](mailto:rna63@drexel.edu)

## Project Overview

This project explores the influence of different features on the listing price of used cars,
and examines the efficacy of multiple machine learning regression models for predicting
the vehicle price. The trained model is used in a web application which can interactively
return a price based on vehicle data input by the user.

## Dataset

We located and considered various datasets. This dataset chosen was based on data from Craigslist used car sales listings.

The dataset is available on Kaggle at <https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data>

The columns include price, condition, manufacturer, and latitude/longitude plus 18 other categories.

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

The app is also deployed online at <https://ranton256-used-car-sale-price--streamlit-price-predictor-tt183q.streamlit.app/>.

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
   2. activate the environment: on Mac/Linux run 'source .venv/bin/activate'
   3. Run 'pip install -r requirements.txt' from the unzipped project directory.
2. [ ] Download dataset from <https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data>
   1. unzip the archive.zip file into the project directory.
3. [ ] Create the datasets/craigslist directory
   1. In the same directory as the Jupyter notebook files:
      1. 'mkdir -p datasets/craigslist'
4. [ ] Move the vehicles.csv file from the zip file to the datasets/craigslist directory
   1. 'mv vehicles.csv datasets/craigslist'
5. [ ] Start the jupyter notebook server.
   1. 'jupyter notebook rna63_project_part2.ipynb'

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

## Price Prediction Using Trained Model

The Jupyter notebook 'used_car_price_predictor.ipynb' contains code which loads the saved model,
recreates the same train/test split, and then runs the test set through the model and outputs
the same statistics as the training notebook.

Update the cell that assigns the filenames for the preprocessor, models, and cleaned dataset
to match the outputs from the other notebooks.

The cell should look something like this:

    preprocessor_path = 'model_preprocessor_f_2023_03_13_14_22_04.pkl'
    model_path = "model_xgboost_2023_03_13_16_35_43.pkl"
    dataset_path = "craigslist_full_cleaned_2023_03_12_10_45_22.csv"

This is just a demonstration of how to load and use the saved model, and a sanity check that
it produces the same results as the training notebook on the same data.

## Interactive Price Prediction Using Streamlit App

The Streamlit app contained in 'streamlit_price_predictor.py' allows entering the input
values in a web interface and generating a predicted price using the saved model.

Update the assigned values for the preprocessor_path, model_path, and dataset_path values as you did
for the 'used_car_price_predictor.ipynb' notebook.

The app can be run locally with:

    python -mstreamlit run streamlit_price_predictor.py

When run locally, if no environment variables containing AWS access key credentials are
detected the application will look for the model and dataset files locally.

The app is also deployed to a Streamlit community cloud app.

The deployed cloud app uses model files written to an S3 bucket since they are too large to check into the GitHub repository.

You can use the app at <https://ranton256-used-car-sale-price--streamlit-price-predictor-tt183q.streamlit.app/>

## References

- Chen, T., He, T., Benesty, M., Khotilovich, V., Tang, Y., Cho, H., ... & Zhou, T. (2015). Xgboost: extreme gradient boosting. R package version 0.4-2, 1(4), 1-4.
- Coefficient of determination. (Mar 2023). In _Wikipedia_. <https://en.wikipedia.org/wiki/Coefficient_of_determination>
- Ho, T. K. (1995, August). Random decision forests. In Proceedings of 3rd international conference on document analysis and recognition (Vol. 1, pp. 278-282). IEEE.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1), 55-67.
- Reese, A., (2021). Used Cars Dataset: Vehicles listings from Craigslist.org, v10. Retrieved Jan 28, 2023 from <https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data>.
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.
