import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

ft_features = pd.read_csv('ft_features_combined_iso_11_05.csv')
ft_properties = pd.read_csv('ft_properties_combined_iso_03_05.csv')

feature_column_rename = {'TOS [h]': 'TOS (h)', 'p(PHG) [bar]': 'p(PHG) (bar)', 'GHSV_PV [h-1]':'GHSV_PV (h-1)'}
ft_features.rename(feature_column_rename, axis=1, inplace=True)

#ft_features.drop(['%Mn','PSD', 'Ratio_H2/CO_in'], axis=1, inplace=True)

dtreg = DecisionTreeRegressor
rfreg = RandomForestRegressor
gbreg = GradientBoostingRegressor
knnreg = KNeighborsRegressor
lr = LinearRegression
xgbreg = XGBRegressor

model_name_list = ['random_forest', 'gradient_boost', 'decision_tree', 'XGB']#'KNN']
model_instance_list = [rfreg, gbreg, dtreg, xgbreg, knnreg]

exclude_targets = ['S_C10+_paraffins', 'S_C10+_olefins', 'S_C10+_OH']

# Use the best model for training each of the properties
for i, model_name in enumerate(model_name_list):
    print(f'RUNNING MODEL {model_name}')
    train_r2_scores = []
    test_r2_scores = []
    train_mae_scores = []
    test_mae_scores = []
    columns = []
    X = ft_features
    with open(f'models/{model_name}_best_parameters_combined_data_iso.pkl','rb') as f:
            best_parameters = pickle.load(f)
    print(best_parameters)
    for col, parameters in best_parameters.items():
        if col in exclude_targets:
            continue
        else:
            
            print(col,'\n')
            y = ft_properties[col]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            print('---------------Running with best parameters--------------')
            if model_name == 'linear':
                model = model_instance_list[i]()
                model.fit(X_train_scaled, y_train)
            else:
                model = model_instance_list[i](**parameters)
                model.fit(X_train, y_train)

            #save models
        #     joblib.dump(model, open(f'models/model_random_forest_best_parameter_{col}.sav','wb'))

            with open(f'models/model_{model_name}_best_parameter_{col}.pkl','wb') as f:
                pickle.dump(model,f)
    #         with open(f'models/model_{model_name}_best_parameter_{col}.pkl', 'rb') as f:
    #             model = pickle.load(f)

            y_pred = model.predict(X_train)
            # test the models on already seen data
            r2 = r2_score(y_train, y_pred)
    #         print('r2 for train: ', r2)
            mae = mean_absolute_error(y_train, y_pred)
    #         print('MAE for train: ', mae)
            mse = mean_squared_error(y_train, y_pred)
    #         print('RMSE for train: ', mse**(1/2), '\n')

            train_r2_scores.append(r2)
            train_mae_scores.append(mae)

            y_pred = model.predict(X_test)
            # test the models for prediction on unseen data
            r2 = r2_score(y_test, y_pred)
    #         print('r2 for test: ', r2)
            mae = mean_absolute_error(y_test, y_pred)
    #         print('MAE for test: ', mae)
            mse = mean_squared_error(y_test, y_pred)
    #         print('RMSE for test: ', mse**(1/2), '\n')

            test_r2_scores.append(r2)
            test_mae_scores.append(mae)
            columns.append(col)
    
    plt.rcParams.update({
    'font.size': 14,          # base font
    'axes.titlesize': 16,     # subplot titles
    'axes.labelsize': 14,     # x/y labels
    'xtick.labelsize': 12,    # x tick labels
    'ytick.labelsize': 12,    # y tick labels
    'legend.fontsize': 12})

    
    x = np.arange(len(columns[:-3]))
    width = 0.35

    fig = plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_r2_scores[:-3], width, label='Train $R^2$')
    plt.bar(x + width/2, test_r2_scores[:-3], width, label=r'Test $R^2$')
    plt.axhline(y=0.9, linestyle='--', linewidth=2, label='$R^2$ = 0.9')

    plt.xticks(x, columns[:-3], rotation=45)
    plt.ylabel('$R^2$ Score')
    plt.title(f'{model_name}')
    plt.legend(bbox_to_anchor=(1, 1))

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'plots/r2_plot_for_each_property_{model_name}.jpg')
    fig.clear()

    fig = plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, train_mae_scores[:-3], width, label='Train MAE')
    plt.bar(x + width/2, test_mae_scores[:-3], width, label='Test MAE')
    #plt.axhline(y=0.9, linestyle='--', linewidth=2, label='R2 = 0.9')

    plt.xticks(x, columns[:-3], rotation=45)
    plt.ylabel('MAE')
    plt.title(f'{model_name}')
    plt.legend(bbox_to_anchor=(1, 1))

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'plots/mae_plot_for_each_property_{model_name}.jpg')
    fig.clear()