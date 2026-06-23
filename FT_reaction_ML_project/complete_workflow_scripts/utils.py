import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def processed_data(features_csv, properties_csv, features_columns_drop=None, properties_columns_drop=None):
    features = pd.read_excel(features_csv)
    properties = pd.read_excel(properties_csv)
    if features_columns_drop is not None:
        features.drop(features_columns_drop, axis=1, inplace=True)
    if properties_columns_drop is not None:
        properties.drop(properties_columns_drop, axis=1, inplace=True)
    return features, properties