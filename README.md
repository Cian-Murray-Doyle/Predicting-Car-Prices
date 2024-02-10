# Using Supervised Machine Learning for Price Predictions

## About

Akashi is a Japanese car company that has recently recovered from a failed attempt to launch vehicles in the US market. During this attempt, sales were low as consumers found the cars to be overpriced. Executives had set price estimates based on previous experience setting prices in the Japanese market. This process relied heavily on unquantifiable intuition and assumed that factors influencing prices in Japan would be the same in the US.

Akashi’s management team has asked you to build a regression model based on a dataset of cars for sale in the American market. Their hope is that they’ll be able to use this model to predict the most appropriate price for the cars they’ll put on the American market.

Skills Showcased

-   Supervised Machine Learning - Decision Trees, KNN, SVR
-   Data Cleaning
-   Feature Engineering
-   Data Analysis

[View more projects like this!](https://cian-murray-doyle.github.io/)

## Libraries Overview

The following Python libraries will be used for this project.

``` python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as skl 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

## Preparing the Data

Before using the regression models the dataset needs to be prepared. First we will define our x/predictors and y/response variables.

``` python
response = us_car_prices[["price"]]
predictors = us_car_prices.drop(["price"],axis=1)
```

Next we will look at removing features that are measuring the same or similar metrics, for example we can drop `"carwidth"` and `"carheight"` and keep `"carlength"`.

![](images/data_vis.PNG)
