## House Price Prediction

This repository contains code and data for predicting house prices using machine learning techniques.

## Dataset

The dataset used in this project is the House Prices: Advanced Regression Techniques from Kaggle. It consists of 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The goal is to predict the final price of each home.

## Dependencies

The code is written in Python 3. The following Python libraries are required:

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* 
## Usage

The main code is in the house_price_prediction.ipynb Jupyter notebook. The notebook contains the following sections:

1 Exploratory Data Analysis (EDA) - visualization and statistical analysis of the dataset

2 Data Preprocessing - handling missing values, encoding categorical variables, feature scaling

3 Feature Selection - selecting the most important features for the prediction task

4 Model Selection - trying out different regression models and comparing their performance

5 Model Evaluation - evaluating the selected model on the test set and analyzing the results

6 Model Deployment - using the selected model to predict house prices on new data


## Results

The best performing model on the test set was the XGBoost regressor with an RMSE of 0.122. The most important features for the prediction task were the overall quality of the house, the total square footage, and the number of cars that can be parked in the garage.

## Conclusion

In this project, we have demonstrated how to predict house prices using machine learning techniques. We have shown that it is possible to achieve good performance on this task by carefully selecting features and tuning the hyperparameters of the chosen model.






