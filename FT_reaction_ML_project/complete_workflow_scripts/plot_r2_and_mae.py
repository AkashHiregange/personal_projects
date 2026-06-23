import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

def plot_performance(features, properties, model_instance_list, model_name_list, 
                     exclude_targets=None, parity=False):
    
    plt.rcParams.update({
        'font.size': 14,          # base font
        'axes.titlesize': 16,     # subplot titles
        'axes.labelsize': 14,     # x/y labels
        'xtick.labelsize': 12,    # x tick labels
        'ytick.labelsize': 12,    # y tick labels
        'legend.fontsize': 12})
    
    # Use the best model for training each of the properties
    for i, model_name in enumerate(model_name_list):
        print(f'RUNNING MODEL {model_name}')
        train_r2_scores = []
        test_r2_scores = []
        train_mae_scores = []
        test_mae_scores = []
        columns = []
        X = features
        with open(f'models/{model_name}_best_parameters.pkl','rb') as f:
                best_parameters = pickle.load(f)
        print(best_parameters)
        for col, parameters in best_parameters.items():
            if exclude_targets is not None:
                if col in exclude_targets:
                    continue
            else:

                print(col,'\n')
                y = properties[col]
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                print('---------------Running with best parameters--------------')
                
                model = model_instance_list[i](**parameters)
                model.fit(X_train, y_train)
                
                with open(f'models/model_{model_name}_best_parameter_{col}.pkl','wb') as f:
                    pickle.dump(model,f)

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

        x = np.arange(len(columns))
        width = 0.35

        fig = plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, train_r2_scores, width, label='Train $R^2$')
        plt.bar(x + width/2, test_r2_scores, width, label=r'Test $R^2$')
        plt.axhline(y=0.9, linestyle='--', linewidth=2, label='$R^2$ = 0.9')

        plt.xticks(x, columns, rotation=45)
        plt.ylabel('$R^2$ Score')
        plt.title(f'{model_name}')
        plt.legend(bbox_to_anchor=(1, 1))

        plt.tight_layout()
        #plt.show()
        plt.savefig(f'plots/r2_plot_for_each_property_{model_name}.jpg')
        fig.clear()

        fig = plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, train_mae_scores, width, label='Train MAE')
        plt.bar(x + width/2, test_mae_scores, width, label='Test MAE')
        #plt.axhline(y=0.9, linestyle='--', linewidth=2, label='R2 = 0.9')

        plt.xticks(x, columns, rotation=45)
        plt.ylabel('MAE')
        plt.title(f'{model_name}')
        plt.legend(bbox_to_anchor=(1, 1))

        plt.tight_layout()
        #plt.show()
        plt.savefig(f'plots/mae_plot_for_each_property_{model_name}.jpg')
        fig.clear()