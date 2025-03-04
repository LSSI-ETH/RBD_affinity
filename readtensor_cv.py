import pandas as pd
import numpy as np
import os
import math
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
from utils import k_fold_cv, record_metrics
import gc  # Garbage collector
import argparse
from sklearn.svm import SVR

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

# ---------------------
# Model Toggles
use_gp = True
use_lr = True
use_mlp = True
use_ridge = True
use_svm = True
use_elastic_net = True
use_random_forest = True
use_xgboost = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

now = datetime.now()
print(now)

data_path = f'/cluster/home/jiahan/SSHPythonProject/antibody_kd_regression/data/data_EPFL.csv'


tensor_str = f"{task_name}_{embedding}_layer{esm_layer}_{pool_method}pool{pool_axis}"
embed_str = f"{embedding}_layer{esm_layer}_{pool_method}pool{pool_axis}"
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

df = pd.read_csv(data_path)
y_all = df['Mean_Relative_Affinity'].values

X_all = torch.load(f'/cluster/project/reddy/jiahan/{tensor_str}_tensor.pt')
print("read X in")
print(X_all.shape)

# Initialize metrics dataframe
metrics_df = pd.DataFrame()

for antibody in unique_antibodies:
    print(f'now doing {antibody}')
    mask = (df['Antibody'] == antibody).values
    X = X_all[mask]
    y = y_all[mask]
    mutations = Mutations_all[mask]

    # Initialize polynomial features for Ridge and ElasticNet
    poly = PolynomialFeatures(degree=2)

    # Dictionary to store metrics for all models
    metrics = {
        'Antibody': [],
        'Model': [],
        'MSE_Mean': [], 'MSE_Std': [],
        'Pearson_Mean': [], 'Pearson_Std': [],
        'Spearman_Mean': [], 'Spearman_Std': [],
        'R2_Mean': [], 'R2_Std': [],
        'Under_Most_Inaccurate': [],
        'Over_Most_Inaccurate': []
    }
    def record_metrics(model_name, results):
        metrics['Antibody'].append(antibody)
        metrics['Model'].append(model_name)
        metrics['MSE_Mean'].append(results['MSE'][0])
        metrics['MSE_Std'].append(results['MSE'][1])
        metrics['Pearson_Mean'].append(results['Pearson Correlation'][0])
        metrics['Pearson_Std'].append(results['Pearson Correlation'][1])
        metrics['Spearman_Mean'].append(results['Spearman Correlation'][0])
        metrics['Spearman_Std'].append(results['Spearman Correlation'][1])
        metrics['R2_Mean'].append(results['R2'][0])
        metrics['R2_Std'].append(results['R2'][1])
        metrics['Under_Most_Inaccurate'].append(results['Under Most Inaccurate'])
        metrics['Over_Most_Inaccurate'].append(results['Over Most Inaccurate'])

   # ----- GP model -----
    if use_gp:
        kernel = ConstantKernel(1.0, (1e-2, 1000)) * RBF(1.0, (1, 50)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-3, 1))
        gp = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=10, normalize_y=True)
        gp_metrics = k_fold_cv(X, y, gp, mutations=mutations)
        record_metrics('GP', gp_metrics)

    # ----- Linear Regression -----
    if use_lr:
        lr_model = LinearRegression()
        lr_metrics = k_fold_cv(X, y, lr_model, mutations=mutations)
        record_metrics('Linear Regression', lr_metrics)

    # ----- MLP -------
    if use_mlp:
        mlp = MLPRegressor(hidden_layer_sizes=(64,), batch_size=64, activation='relu', solver='adam', max_iter=1000, alpha=0.01, learning_rate_init=0.0001, random_state=42)
        mlp_metrics = k_fold_cv(X, y, mlp, mutations=mutations)
        record_metrics('MLP', mlp_metrics)

    # ----- Ridge -----
    if use_ridge:
        ridge_reg = Ridge(alpha=7.05)
        ridge_metrics = k_fold_cv(X, y, ridge_reg, poly=poly, mutations=mutations)
        record_metrics('Ridge', ridge_metrics)

    # ----- ElasticNet -----
    if use_elastic_net:
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
        elastic_net_metrics = k_fold_cv(X, y, elastic_net, poly=poly, mutations=mutations)
        record_metrics('ElasticNet', elastic_net_metrics)

    # ----- Random Forest -----
    if use_random_forest:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_metrics = k_fold_cv(X, y, rf, mutations=mutations)
        record_metrics('Random Forest', rf_metrics)

    # ----- XGBoost -----
    if use_xgboost:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
        xgb_metrics = k_fold_cv(X, y, xgb, mutations=mutations)
        record_metrics('XGBoost', xgb_metrics)
    
    if use_svm:
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svm_metrics = k_fold_cv(X, y, svr, mutations=mutations)
        record_metrics('SVM_rbf', svm_metrics)
        
        svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
        svm_metrics = k_fold_cv(X, y, svr, mutations=mutations)
        record_metrics('SVM_linear', svm_metrics)
    # Append metrics to the consolidated DataFrame
    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics)])

# Save all metrics to a single CSV
metrics_csv_path = os.path.join(plot_dir, "consolidated_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"All metrics saved to {metrics_csv_path}")

gc.collect()
