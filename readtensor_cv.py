import pandas as pd
import numpy as np
import os
import math
import joblib 
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
import torch
import matplotlib.pyplot as plt
from utils import ModelEvaluator_fixfold
import gc  # Garbage collector
import argparse
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from model_definitions import get_model_configs

parser = argparse.ArgumentParser(description='Train GP and MLP models on embeddings.')

# Adding arguments
parser.add_argument('--embedding', type=str, default='onehot', 
                    help='Type of embedding to use.')
parser.add_argument('--mlp_hidden_dim', type=int, nargs='+', default = [64],
                    help='List of hidden layer sizes for the MLP. E.g., --mlp_hidden_dim 64 32')
parser.add_argument('--length_scale', type=float, default = 10,
                    help='Length scale for RBF kernel in GP.')
parser.add_argument('--esm_layer', type=int, default = 15,
                    help='Embedding layer in ESM2.')
parser.add_argument('--pool_method', type=str, default = 'mean',
                    help='Pooling method for PLM embeddings.')
parser.add_argument('--pool_axis', type=int, default = 0,
                    help='Pooling axis for embeddings.')        
parser.add_argument('--y_name', type=str, default = 'Mean_Relative_Affinity',
                    help='End point column name.')   
parser.add_argument('--task_name', type=str, default='moulana',
                    help='Task name.') 


args = parser.parse_args()
embedding = args.embedding
mlp_hidden_dim = tuple(args.mlp_hidden_dim)
length_scale = args.length_scale
esm_layer = args.esm_layer
pool_method = args.pool_method
pool_axis = args.pool_axis
y_name = args.y_name
task_name = args.task_name
use_poly = False
# ---------------------
use_flags = {
    "use_gp": True,
    "use_lr": True,
    "use_mlp": True,
    "use_ridge": True,
    "use_elastic_net": True,
    "use_random_forest": True,
    "use_xgboost": True,
    "use_svm": True
}

#------
param_grids = {
    # "GP": {"alpha": [1e-10, 1e-6, 1e-3]},  # Uncomment if needed and used directly
    "MLP": {
        "mlpregressor__hidden_layer_sizes": [(64,), (128,), (256,)],
        "mlpregressor__learning_rate_init": [0.0001, 0.001, 0.01],
        "mlpregressor__alpha": [0.0001, 0.001, 0.01]
    },
    "Ridge": {
        "ridge__alpha": [0.1, 1, 10, 100]
    },
    "ElasticNet": {
        "elasticnet__alpha": [0.001, 0.01, 0.1],
        "elasticnet__l1_ratio": [0.2, 0.5, 0.8]
    },
    "RandomForest": {
        "randomforestregressor__n_estimators": [100, 1000],
        "randomforestregressor__max_depth": [None, 10, 20]
    },
    "XGBoost": {
        "xgbregressor__n_estimators": [50, 100],
        "xgbregressor__max_depth": [6, 10],
        "xgbregressor__learning_rate": [0.01, 0.1]
    },
    "SVR_rbf": {
        "svr__C": [1,10,100],
        "svr__epsilon": [0.001, 0.01, 0.1],
        "svr__gamma": ["scale", "auto"]
    },
    "SVR_linear": {
        "svr__C": [0.01, 0.1, 1],
        "svr__epsilon": [0.001, 0.01, 0.1]
    }
}

model_configs = get_model_configs(param_grids, use_flags)

#--------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

now = datetime.now()
print(now)

data_path = f'/cluster/home/jiahan/SSHPythonProject/antibody_kd_regression/data/data_EPFL.csv'


tensor_str = f"{task_name}_{embedding}_layer{esm_layer}_{pool_method}pool{pool_axis}"
#embed_str = f"{embedding}_layer{esm_layer}_{pool_method}pool{pool_axis}"
model_str = f"{y_name}_{tensor_str}_mlpdim{mlp_hidden_dim}_gplen{length_scale}"

df = pd.read_csv(data_path)
y_all = df[y_name].values
print(data_path)

unique_antibodies = df['Antibody'].unique()
print(f'unique_antibodies:{unique_antibodies}')

Mutations_all = df['Mutations']

plot_dir = f'{now}_{model_str}_result'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


X_all = torch.load(f'/cluster/project/reddy/jiahan/{tensor_str}_tensor.pt') 
print("read X in")
print(X_all.shape)
folds_all = df['fold_random_5'].values
# Initialize metrics dataframe
metrics_df = pd.DataFrame()

for antibody in unique_antibodies:
    print(f'now doing {antibody}')
    mask = (df['Antibody'] == antibody).values
    X = X_all[mask]
    y = y_all[mask]

    mutations = Mutations_all[mask]
    folds = folds_all[mask]
    
    for cfg in model_configs:
        if cfg["enabled"]:
            print(f"\nNow evaluating: {cfg['name']}")
            poly_features = None
            if use_poly:
                if embedding.lower() == "onehot" and cfg["name"] in ["Linear Regression", "Ridge", "ElasticNet", "SVM_linear"] or \
                embedding.lower() == "16encode" and cfg["name"] in ["Ridge", "ElasticNet"]:
                    print(f'using poly for {embedding}_{cfg["name"]}')
                    poly_features = PolynomialFeatures(degree=2, include_bias=False)
                else:
                    print("Not using poly")
                
            evaluator = ModelEvaluator_fixfold(
                        X=X,
                        y=y,
                        folds=folds,
                        model=cfg["model"],
                        model_name=cfg["name"],
                        plot_dir=plot_dir,
                        antibody=antibody,
                        embedding=embedding,
                        param_grid=cfg["param_grid"],
                        mutations=mutations,
                        poly=poly_features
                    )
            # Set model-specific configs
            metrics = evaluator.evaluate()
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)

# Save all metrics to a single CSV
metrics_csv_path = os.path.join(plot_dir, "cv_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"All metrics saved to {metrics_csv_path}")

gc.collect()
