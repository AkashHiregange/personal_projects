import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from utils import processed_data
from best_parameter_search import search_best_parameters
from plot_r2_and_mae import plot_performance

lr = LinearRegression()
dtreg = DecisionTreeRegressor()
rfreg = RandomForestRegressor()
gbreg = GradientBoostingRegressor()
knnreg = KNeighborsRegressor()
xgbreg = XGBRegressor()

ft_features, ft_properties = processed_data('features.xlsx', 'properties.xlsx') 

# gradient boost
param_grid_gb = {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4, 6], 'learning_rate': [0.01, 0.05, 0.1], 'subsample':[0.6,0.7,0.9]}
print(search_best_parameters(ft_features, ft_properties, gbreg, param_grid_gb, 'gradient_boost'))

# random forest
# param_grid_rf = {'n_estimators': [100, 200, 250], 'max_depth': [5, 10, 15, 20], 'min_samples_leaf':[5,7,10]}
# print(search_best_parameters(ft_features, ft_properties, rfreg, param_grid_rf, 'random_forest'))

# # decision_tree
# param_grid_dt = {'max_depth': [2, 5, 10, 15], 'min_samples_leaf':[5,7,10], 'splitter':['best', 'random'],'min_samples_split':[10,20,30]}
# print(search_best_parameters(ft_features, ft_properties, dtreg, param_grid_dt, 'decision_tree'))

# # XGB
# param_grid_xgb = {'n_estimators': [100, 200, 250], 'max_depth': [6, 10, 15], 'learning_rate':[0.01,0.05,0.1]}
# print(search_best_parameters(ft_features, ft_properties, xgbreg, param_grid_xgb, 'XGB'))

dtreg = DecisionTreeRegressor
rfreg = RandomForestRegressor
gbreg = GradientBoostingRegressor
knnreg = KNeighborsRegressor
lr = LinearRegression
xgbreg = XGBRegressor

model_name_list = ['gradient_boost']#'KNN']
model_instance_list = [gbreg]

exclude_targets = []

plot_performance(ft_features, ft_properties, model_instance_list, model_name_list)