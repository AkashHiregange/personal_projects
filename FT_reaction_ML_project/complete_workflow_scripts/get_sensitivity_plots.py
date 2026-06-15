import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import shap

ft_features = pd.read_csv('ft_features_combined_iso_11_05.csv')
ft_properties = pd.read_csv('ft_properties_combined_iso_03_05.csv')
feature_column_rename = {'TOS [h]': 'TOS (h)', 'p(PHG) [bar]': 'p(PHG) (bar)', 'GHSV_PV [h-1]':'GHSV_PV (h-1)'}
ft_features.rename(feature_column_rename, axis=1, inplace=True)
# ft_features.drop(['TOS (h)'], axis=1, inplace=True)

ft_features['Catalyst_form_pellet'] = ft_features['Catalyst_form_pellet'].astype(int)
ft_features['Catalyst_form_powder'] = ft_features['Catalyst_form_powder'].astype(int)
feature_list = list(ft_features.columns)
print(feature_list)

model_name = 'XGB'
X = ft_features
with open(f'models/{model_name}_best_parameters_combined_data_iso.pkl','rb') as f:
    best_parameters = pickle.load(f)

for col, parameters in best_parameters.items():
    print(col,'\n')
    y = ft_properties[col]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    model = XGBRegressor(**parameters)
    model.fit(X_train, y_train)
    
    X_train_df = pd.DataFrame(X_train,columns=ft_features.columns)
    explainer = shap.Explainer(model, X_train_df)
    shap_values = explainer(X_train_df)

    importance = np.abs(shap_values.values).mean(axis=0)

    importance_series = pd.Series(importance, index=X.columns)

    top_feature = importance_series.idxmax()
    top_features = importance_series.sort_values(ascending=False).head(4).index.tolist()
    print(top_features)
    if 'Catalyst_form_pellet' in top_features:
        top_features.remove('Catalyst_form_pellet')
    if 'Catalyst_form_powder' in top_features:
        top_features.remove('Catalyst_form_powder')
    print(top_features)
    for feature in top_features:
        grid = np.linspace(X_train[feature].min(),X_train[feature].max(),20)
        X_sample = X_train.sample(50, random_state=42).copy()
        ice_curves = []

        for i in range(len(X_sample)):
            x = X_sample.iloc[i].copy()

            preds = []

            for value in grid:
                x[feature] = value
                preds.append(model.predict([x])[0])

            ice_curves.append(preds)

        ice_curves = np.array(ice_curves)
        ice_mean = ice_curves.mean(axis=0)
        ice_std = ice_curves.std(axis=0)

        fig = plt.figure(figsize=(8,6))
        for curve in ice_curves:
            plt.plot(grid, curve, color="gray", alpha=0.2)
        plt.plot(grid, ice_mean, color="blue", linewidth=3, label="Mean ICE")

        plt.fill_between(grid,ice_mean - ice_std,ice_mean + ice_std,color="blue",alpha=0.2,
                         label=f"std: {np.round(np.mean(ice_std))}")

        plt.xlabel(f"Feature {feature} ")
        plt.ylabel(f"Predicted {col}")
        plt.legend()
        if feature == 'Ratio_H2/CO_in':
            feature = 'Ratio_H2-CO_in'
        fig.savefig(f'plots/Sensitivity plot {col} {feature}.jpg')
        fig.clear()