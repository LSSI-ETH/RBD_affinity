from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# This will be overridden from benchmarking script
param_grids = {}

def get_model_configs(param_grids, use_flags, random_state=0):
    return [
        {
            "name": "GP",
            "enabled": use_flags.get("use_gp", False),
            "model": GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-1, 100)) +
                        WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1)),
                #kernel = ConstantKernel(1.0, (1e-2, 1000)) * RBF(1.0, (1, 50)) + 
                #WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1)),
                normalize_y=True,
                n_restarts_optimizer=50,
                random_state=random_state
            ),
            "param_grid": None
        },
        {
            "name": "Linear Regression",
            "enabled": use_flags.get("use_lr", False),
            "model": LinearRegression(),
            "param_grid": None
        },
        {
            "name": "MLP",
            "enabled": use_flags.get("use_mlp", False),
            "model": MLPRegressor(hidden_layer_sizes=(64,), batch_size=64, activation='relu',
                                  solver='adam', max_iter=1000, alpha=0.01,
                                  learning_rate_init=0.0001, random_state=random_state),
            "param_grid": param_grids.get("MLP", {})
        },
        {
            "name": "Ridge",
            "enabled": use_flags.get("use_ridge", False),
            "model": Ridge(alpha=7.05),
            "param_grid": param_grids.get("Ridge", {})
        },
        {
            "name": "ElasticNet",
            "enabled": use_flags.get("use_elastic_net", False),
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, random_state=random_state),
            "param_grid": param_grids.get("ElasticNet", {})
        },
        {
            "name": "Random Forest",
            "enabled": use_flags.get("use_random_forest", False),
            "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state),
            "param_grid": param_grids.get("RandomForest", {})
        },
        {
            "name": "XGBoost",
            "enabled": use_flags.get("use_xgboost", False),
            "model": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=random_state),
            "param_grid": param_grids.get("XGBoost", {})
        },
        {
            "name": "SVM_rbf",
            "enabled": use_flags.get("use_svm", False),
            "model": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "param_grid": param_grids.get("SVR_rbf", {})
        },
        {
            "name": "SVM_linear",
            "enabled": use_flags.get("use_svm", False),
            "model": SVR(kernel='linear', C=1.0, epsilon=0.1),
            "param_grid": param_grids.get("SVR_linear", {})
        },
    ]
