# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:50:35 2023

@author: cian3
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as skl 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

us_car_prices = pd.read_csv("us_car_prices.csv",index_col=0)
us_car_prices.head()

response = us_car_prices[["price"]]
predictors = us_car_prices.drop(["price"],axis=1)

numeric_columns = us_car_prices.iloc[:,10:]
numeric_columns = numeric_columns.drop(
    ["symboling","stroke","compressionratio","peakrpm",
     "horsepower","carlength","carheight","citympg"],axis=1)

correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix,vmin=-1.0,vmax=1.0,annot=True)

predictors = predictors.drop(["symboling","stroke","compressionratio",
                              "peakrpm","horsepower","carlength",
                              "carwidth","carheight","citympg"],axis=1)

predictors = pd.get_dummies(predictors,drop_first=True)

linear_regression = skl.LinearRegression()
linear_regression.fit(predictors,response)
print(linear_regression.coef_)
print(linear_regression.intercept_)

r_squared = linear_regression.score(predictors,response)
print("R_squared: ",r_squared)

n = len(response)
k = predictors.shape[1]
adjusted_r_squared = 1-((1-r_squared)*(n-1)/(n-k-1))
print("Adjusted R_squared: ",adjusted_r_squared)

response_predictions = linear_regression.predict(predictors)
residuals = response - response_predictions
print(residuals.mean())
print(residuals.std())

plt.hist(residuals,bins=100)
plt.show()

plt.scatter(response_predictions,residuals)
plt.title("Homoscedasticity")
plt.xlabel("Response Predictions")
plt.ylabel("Residuals")
plt.show()

predictors_train, predictors_test, response_train, response_test = train_test_split(
    predictors, response, test_size=0.2, random_state=42)

tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(predictors_train,response_train)
tree_predictions = tree_regressor.predict(predictors_test)

tree_mae = mean_absolute_error(response_test,tree_predictions)
tree_mse = mean_squared_error(response_test,tree_predictions)
tree_r_squared = r2_score(response_test,tree_predictions)
print("Decision Tree Mean Absolute Error: ",tree_mae)
print("Decision Tree Mean Squared Error:" ,tree_mse)
print("Decision Tree R squared: ",tree_r_squared)

knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(predictors_train,response_train)
knn_predictions = knn_regressor.predict(predictors_test)

knn_mae = mean_absolute_error(response_test,knn_predictions)
knn_mse = mean_squared_error(response_test,knn_predictions)
knn_r_squared = r2_score(response_test,knn_predictions)
print("KNN Mean Absolute Error: ",knn_mae)
print("KNN Mean Squared Error: ",knn_mse)
print("KNN R squared: ",knn_r_squared)

svr_regression = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_regression.fit(predictors_train, response_train["price"])
svr_predictions = svr_regression.predict(predictors_test)

svr_mae = mean_absolute_error(response_test,svr_predictions)
svr_mse = mean_squared_error(response_test,svr_predictions)
svr_r_squared = r2_score(response_test,svr_predictions)
print("SVR Mean Absolute Error: ",svr_mae)
print("SVR Mean Squared Error: ",svr_mse)
print("SVR R squared: ",svr_r_squared)
