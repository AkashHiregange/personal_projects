def processed_data(features_csv, properties_csv, features_columns_drop=None, properties_columns_drop=None):
    features = pd.read_excel(features_csv)
    properties = pd.read_excel(properties_csv)
    if features_columns_drop is not None:
        features.drop(features_columns_drop, axis=1, inplace=True)
    if properties_columns_drop is not None:
        properties.drop(properties_columns_drop, axis=1, inplace=True)
    return features, properties

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