# README - Used Car Price Regression

## Team

- Project Grp 08

### Team Members

- Team member 1
  - Name: Richard Anton
  - Email: [rna63@drexel.edu](mailto:rna63@drexel.edu)


# TODO:  thorough README document describing how your code runs and should be engaged.


## Dataset

Craigslist: <https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data>

This dataset was created by the provider scraping data from Craigslist car listings.
The columns include price, condition, manufacturer, and latitude/longitude plus 18 other categories.

## Geospatial data

The US State geo boundary data was obtained from:

<https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html>

## Notebooks

### Data Preperation Notebook - rna63_project_part2.ipynb

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
a used vehicle.

### TODO: add the geo by state diagram

- [ ] TODO; add streamlist version of inference if possible.
