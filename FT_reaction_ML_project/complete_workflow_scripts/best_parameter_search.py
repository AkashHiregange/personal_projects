import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from utils import processed_data

def search_best_parameters(features, properties, model, param_grid, model_name):
    print(model_name)
    X = features
    from sklearn.model_selection import GridSearchCV
    best_parameters = {}
    for i, col in enumerate(properties.columns):
        print(f'training for {col}\n')
        y = properties[col]
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
    with open(f'models/{model_name}_best_parameters.pkl','wb') as f:
        pickle.dump(best_parameters,f)
    return best_parameters
 
