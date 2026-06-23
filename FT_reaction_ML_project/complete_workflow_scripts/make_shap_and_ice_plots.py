import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def make_shap_plots(features, properties, model_instance, model_name, bar=True):
    
    plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300})
    
    with open(f'models/{model_name}_best_parameters.pkl','rb') as f:
        best_parameters = pickle.load(f)
    print(best_parameters)
    for col, parameters in best_parameters.items():
        print(col,'\n')

        y = properties[col]
        X = features

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
        X_train_df = pd.DataFrame(X_train,columns=features.columns)

        model = model_instance(**parameters)
        model.fit(X_train, y_train)

        explainer = shap.Explainer(model, X_train_df)

        shap_values = explainer(X_train_df, check_additivity=False)
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.xlabel('SHAP value (impact on model output)')
        plt.title(f"SHAP Beeswarm ({col})", pad=15)
        plt.tight_layout()
        plt.savefig(f'plots/SHAP_beeswarm_{model_name}_{col}.jpg', bbox_inches='tight')
        plt.close()

        # Bar plot
        if bar==True:
            plt.figure()
            shap.summary_plot(shap_values, X_train_df, plot_type="bar", show=False)
            plt.xlabel('Mean absolute SHAP value')
            plt.title(f"Feature Importance ({col})", pad=15)
            plt.tight_layout()
            plt.savefig(f'plots/SHAP_bar_{model_name}_{col}.jpg', bbox_inches='tight')
            plt.close()
            
def make_ice_plots_for_top_features(features, properties, model_instance, model_name, n_top_features=2):
    
    plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300})
    
    with open(f'models/{model_name}_best_parameters.pkl','rb') as f:
        best_parameters = pickle.load(f)

    for col, parameters in best_parameters.items():
        y = properties[col]
        X = features
        print(col,'\n')
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

        model = model_instance(**parameters)
        model.fit(X_train, y_train)

        X_train_df = pd.DataFrame(X_train,columns=features.columns)
        explainer = shap.Explainer(model, X_train_df)
        shap_values = explainer(X_train_df, check_additivity=False)

        importance = np.abs(shap_values.values).mean(axis=0)

        importance_series = pd.Series(importance, index=X.columns)

        top_feature = importance_series.idxmax()
        top_features = importance_series.sort_values(ascending=False).head(n_top_features).index.tolist()
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
                    preds.append(model.predict(pd.DataFrame([x]))[0])

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