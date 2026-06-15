import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import shap

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

ft_features = pd.read_csv('ft_features_combined_iso_11_05.csv')
ft_properties = pd.read_csv('ft_properties_combined_iso_03_05.csv')

feature_column_rename = {'TOS [h]': 'TOS', 'p(PHG) [bar]': 'Total Pressure', 'GHSV_PV [h-1]':'GHSV', 'Ratio_H2/CO_in': 'Ratio H2/CO', 'T-Rx':'Temperature'}
ft_features.rename(feature_column_rename, axis=1, inplace=True)
# ft_features.drop(['TOS (h)'], axis=1, inplace=True)

ft_features['Catalyst_form_pellet'] = ft_features['Catalyst_form_pellet'].astype(int)
ft_features['Catalyst_form_powder'] = ft_features['Catalyst_form_powder'].astype(int)
# ft_features = ft_features[['%Mn', 'Ratio_H2/CO_in', 'Catalyst_form_pellet', 'Catalyst_form_powder']]
print(list(ft_features.columns))

model_name = 'XGB'
with open(f'models/{model_name}_best_parameters_combined_data_iso.pkl','rb') as f:
    best_parameters = pickle.load(f)
print(best_parameters)
for col, parameters in best_parameters.items():
    print(col,'\n')

    y = ft_properties[col]
    X = ft_features

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
    X_train_df = pd.DataFrame(X_train,columns=ft_features.columns)
    
    model = XGBRegressor(**parameters)
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train_df)

    shap_values = explainer(X_train_df)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.xlabel('SHAP value (impact on model output)')
    plt.title(f"SHAP Beeswarm ({col})", pad=15)
    plt.tight_layout()
    plt.savefig(f'SHAP_beeswarm_{model_name}_{col}.jpg', bbox_inches='tight')
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_train_df, plot_type="bar", show=False)
    plt.xlabel('Mean absolute SHAP value')
    plt.title(f"Feature Importance ({col})", pad=15)
    plt.tight_layout()
    plt.savefig(f'SHAP_bar_{model_name}_{col}.jpg', bbox_inches='tight')
    plt.close()
        
        
