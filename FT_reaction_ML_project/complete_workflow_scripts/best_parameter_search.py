import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

ft_features = pd.read_csv('ft_features_combined_iso_11_05.csv')
ft_properties = pd.read_csv('ft_properties_combined_iso_03_05.csv')

#ft_features = ft_features[['%Mn', 'Ratio_H2/CO_in', 'Catalyst_form_pellet', 'Catalyst_form_powder']]
#ft_features.drop(['%Mn','PSD', 'Ratio_H2/CO_in'], axis=1, inplace=True)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

lr = LinearRegression()
dtreg = DecisionTreeRegressor()
rfreg = RandomForestRegressor()
gbreg = GradientBoostingRegressor()
knnreg = KNeighborsRegressor()
xgbreg = XGBRegressor()

X = ft_features
def search_best_parameters(model, param_grid, model_name):
    print(model_name)
    from sklearn.model_selection import GridSearchCV
    best_parameters = {}
    for i, col in enumerate(ft_properties.columns):
        print(f'training for {col}\n')
        y = ft_properties[col]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        #from sklearn.preprocessing import StandardScaler
        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train)
        #X_test_scaled = scaler.transform(X_test)
#         param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [2, 10, 20], 'learning_rate': [0.1, 0.5, 1]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_parameters[col] = grid_search.best_params_
        print(f"Best parameters for {col}: ", grid_search.best_params_)
    with open(f'models/{model_name}_best_parameters_combined_data_iso.pkl','wb') as f:
        pickle.dump(best_parameters,f)
    return best_parameters

# gradient boost
#param_grid_gb = {'n_estimators': [100, 200, 300], 'max_depth': [2, 3, 4, 6], 'learning_rate': [0.01, 0.05, 0.1], 'subsample':[0.6,0.7,0.9]}
#print(search_best_parameters(gbreg, param_grid_gb, 'gradient_boost'))

#random forest
#param_grid_rf = {'n_estimators': [100, 200, 250], 'max_depth': [5, 10, 15, 20], 'min_samples_leaf':[5,7,10]}
#print(search_best_parameters(rfreg, param_grid_rf, 'random_forest'))

#decision_tree
#param_grid_dt = {'max_depth': [2, 5, 10, 15], 'min_samples_leaf':[5,7,10], 'splitter':['best', 'random'],'min_samples_split':[10,20,30]}
#print(search_best_parameters(dtreg, param_grid_dt, 'decision_tree'))

#XGB
param_grid_xgb = {'n_estimators': [100, 200, 250], 'max_depth': [6, 10, 15], 'learning_rate':[0.01,0.05,0.1]}
print(search_best_parameters(xgbreg, param_grid_xgb, 'XGB'))

#KNN
#param_grid_knn = {'n_neighbors': [10,15,20,25], 'p':[1,2,3], 'weights':['distance']}
#print(search_best_parameters(knnreg, param_grid_knn, 'KNN'))

#linear
#param_grid_lr = {'fit_intercept': [True], 'tol':[1e-6]}
#print(search_best_parameters(lr, param_grid_lr, 'linear'))
 
 
