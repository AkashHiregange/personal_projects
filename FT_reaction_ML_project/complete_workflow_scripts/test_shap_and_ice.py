import utils
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from make_shap_and_ice_plots import make_shap_plots, make_ice_plots_for_top_features

ft_features, ft_properties = utils.processed_data('features.xlsx', 'properties.xlsx') 

model_name = 'gradient_boost'
gbreg = GradientBoostingRegressor
make_shap_plots(ft_features, ft_properties, gbreg, model_name)

make_ice_plots_for_top_features(ft_features, ft_properties, gbreg, model_name)