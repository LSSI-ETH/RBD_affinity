import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
#import umap
import pytorch_lightning as pl
from Bio.Align.Applications import MuscleCommandline
from Bio import AlignIO
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GridSearchCV
from Bio.Seq import Seq
from Bio.Data import CodonTable
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ShuffleSplit


def one_hot_encode_sequences(seq_list, aa_list = 'ACDEFGHIKLMNPQRSTVWYZX-'):
    
    aa_to_index = {aa: index for index, aa in enumerate(aa_list)}
    
    encoded_list = []
    
    for seq in seq_list:
        # Initialize the encoded array for this sequence
        encoded_seq = np.zeros((len(seq), len(aa_list)))
        
        for i, aa in enumerate(seq):
            if aa in aa_to_index:  # This checks also handles unexpected characters by ignoring them
                encoded_seq[i, aa_to_index[aa]] = 1
            else:
                # Handle unexpected characters (e.g., 'X', 'B', 'Z', or other non-standard amino acids)
                # For simplicity, they are not encoded; alternatively, you could add extra dimensions to handle them
                pass
        
        encoded_list.append(encoded_seq)
    
    return encoded_list
    

def pool_embeddings(embeddings, pool_method='mean', axis=0, n_components=2):
    """
    Apply specified pooling method to embeddings including mean, max, min, PCA, and UMAP.
    
    Parameters:
    embeddings (torch.Tensor or np.array): The embeddings to pool. Expected to be a torch.Tensor for mean, max, min; np.array for PCA and UMAP.
    pool_method (str): The pooling method ('mean', 'max', 'min', 'pca', 'umap').
    axis (int): The axis over which to perform the pooling for mean, max, min.
    n_components (int): Number of components to keep for PCA and UMAP.
    
    Returns:
    np.array or torch.Tensor: The pooled embeddings.
    """
    sequence_length, embedding_length = embeddings.shape
    if pool_method in ['mean', 'max', 'min']:
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings)
        if pool_method == 'mean':
            return torch.mean(embeddings, dim=axis)
        elif pool_method == 'max':
            return torch.max(embeddings, dim=axis).values
        elif pool_method == 'min':
            return torch.min(embeddings, dim=axis).values
    elif pool_method in ['pca', 'umap']:
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.view(-1, embedding_length).numpy()  # Flatten and convert to numpy
        else:
            embeddings_np = embeddings.reshape(-1, embeddings.shape[-1])  # Just flatten if already numpy

        # Dimensionality reduction
        if pool_method == 'pca':
            pca = PCA(n_components=n_components)
            reduced_embeddings_np = pca.fit_transform(embeddings_np)
        #elif pool_method == 'umap':
            #reducer = umap.UMAP(n_components=n_components)
            #reduced_embeddings_np = reducer.fit_transform(embeddings_np)

        # Reshape back: (num_sequences, n_components * embedding_length)
        # Note: Adjustments may be needed depending on how you want to structure the output
        flattened_embeddings = reduced_embeddings_np.reshape(-1)
        
        return torch.tensor(flattened_embeddings, dtype=torch.float)

    else:
        raise ValueError(f"Unknown pool_method: {pool_method}")


class EmbeddingAutoencoder(pl.LightningModule):
    def __init__(self, embedding_dim, bottleneck_dim, train_dataloader, val_dataloader, test_dataloader):
        super().__init__()
        self.save_hyperparameters()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
    def train_dataloader(self):
        return self.hparams.train_dataloader

    def val_dataloader(self):
        return self.hparams.val_dataloader

    def test_dataloader(self):
        return self.hparams.test_dataloader
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Unpack the batch tuple
        print(f"Batch type: {type(x)}, shape: {x.shape}")  
        _, decoded = self.forward(x)
        loss = nn.functional.mse_loss(decoded, x)  # Use x for the original data
        self.log("train_loss", loss)
        return loss
    def test_step(self, batch, batch_idx):
        x, _ = batch  # Assuming the test DataLoader is structured like the train DataLoader
        _, decoded = self.forward(x)
        loss = nn.functional.mse_loss(decoded, x)  # Calculate the loss (or other metrics)
        self.log("test_loss", loss)  # Log the loss for visibility in TensorBoard or other loggers
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        
def pool_embeddings_with_autoencoder(autoencoder, embeddings):
    # Ensure autoencoder is in eval mode
    autoencoder.eval()
    
    # Container for pooled embeddings
    pooled_embeddings = []
    
    # Process each embedding
    for emb in embeddings:
        with torch.no_grad():
            # Assuming emb is a tensor. If not, convert it.
            # Add batch dimension [N, C] -> [1, N, C] where N is sequence length and C is channels (embedding_dim)
            emb = emb.unsqueeze(0) 
            # Get the encoded (pooled) representation
            encoded, _ = autoencoder(emb)
            # Remove batch dimension [1, D] -> [D] where D is bottleneck_dim
            pooled_embeddings.append(encoded.squeeze(0))
    
    # Stack pooled embeddings into a single tensor
    return torch.stack(pooled_embeddings)


def align_sequences(aa_sequences):
    """
    Align a list of amino acid sequences using MUSCLE.

    Args:
    aa_sequences (list): A list of string sequences to be aligned.

    Returns:
    list: A list of aligned sequences as strings.
    """
    # Create a temporary file for the input sequences
    input_handle, input_path = tempfile.mkstemp(suffix='.fasta')
    try:
        with os.fdopen(input_handle, 'w') as f:
            for i, seq in enumerate(aa_sequences, 1):
                record = SeqRecord(Seq(seq),
                                   id=f"Seq{i}",
                                   description="")
                SeqIO.write(record, f, "fasta")
        
        # Define the output path
        output_handle, output_path = tempfile.mkstemp(suffix='.fasta')
        os.close(output_handle)  # Close the handle created by mkstemp

        # Path to the MUSCLE executable
        muscle_exe = "muscle"  # Update this path if necessary

        # Run MUSCLE
        muscle_cline = MuscleCommandline(muscle_exe, input=input_path, out=output_path)
        stdout, stderr = muscle_cline()

        # Read the aligned sequences
        alignment = AlignIO.read(output_path, "fasta")

        # Convert the alignment to a list of strings
        aligned_sequences = [str(record.seq) for record in alignment]
    finally:
        # Cleanup temporary files
        os.remove(input_path)
        os.remove(output_path)
    
    return aligned_sequences

# Example usage:
# aa_sequences = ["SEQUENCE1", "SEQUENCE2", "SEQUENCE3"]
# aligned_seqs = align_sequences(aa_sequences)
# print(aligned_seqs)

def plot_metrics(metrics_dict, plot_dir, argstr):
    fig, axs = plt.subplots(1, len(metrics_dict), figsize=(15, 5))
    if len(metrics_dict) == 1:  # If only one metric, ensure axs is iterable
        axs = [axs]
    
    for ax, (metric_name, model_metrics) in zip(axs, metrics_dict.items()):
        for model_name, value in model_metrics.items():
            ax.bar(model_name, value, label=model_name)
        ax.set_title(metric_name)
        ax.set_ylabel('Value')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{argstr}_model_metrics_comparison_plot.png")

def update_metrics_csv(metrics_dict, csv_file_name):
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index').transpose()
    try:
        existing_df = pd.read_csv(csv_file_name, index_col=0)
        updated_df = pd.concat([existing_df, metrics_df])
    except FileNotFoundError:
        updated_df = metrics_df
    
    updated_df.to_csv(csv_file_name)

def hist_y(y):
    plt.hist(y, bins='auto', alpha=0.7, color='blue', edgecolor='black')

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of y')
    # Save the plot as a PNG file
    plt.savefig('y_hist.png')

# Function for up-sampling minority class
def upsample_minority_class(X, y, target_ratio=0.25):
    """Upsample the minority class to meet the specified target ratio."""
    # Calculate class counts
    class_counts = torch.bincount(y)
    majority_class = torch.argmax(class_counts).item()
    minority_class = torch.argmin(class_counts).item()

    # Separate majority and minority class samples
    X_majority = X[y == majority_class]
    X_minority = X[y == minority_class]

    # Calculate target number of samples for the minority class
    target_minority_count = int(len(X_majority) * (target_ratio / (1 - target_ratio)))
    if len(X_minority) < target_minority_count:
        # Upsample the minority class
        X_minority_upsampled = resample(X_minority, replace=True, n_samples=target_minority_count, random_state=42)
        y_minority_upsampled = torch.full((target_minority_count,), minority_class, dtype=torch.long)

        # Concatenate majority and upsampled minority data
        X_upsampled = torch.cat([X_majority, X_minority_upsampled], dim=0)
        y_upsampled = torch.cat([torch.full((len(X_majority),), majority_class, dtype=torch.long), y_minority_upsampled], dim=0)
    else:
        # No up-sampling needed
        X_upsampled, y_upsampled = X, y

    return X_upsampled, y_upsampled

def record_metrics(metrics, model_name, results):
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
    metrics['Best_Hyperparameters'].append(results.get("Best Hyperparameters", "Default"))


# def k_fold_cv(X, y, model, plot_dir, antibody, embedding, param_grid=None, mutations=None):
#     def spearman_corr(y_true, y_pred):
#         return spearmanr(y_true, y_pred).correlation

#     spearman_scorer = make_scorer(spearman_corr, greater_is_better=True)
#     best_params = []
#     model_name = model.__class__.__name__

#     # Wrap with StandardScaler unless it's already in a pipeline
#     if not isinstance(model, Pipeline):
#         model = make_pipeline(StandardScaler(), model)

#     # Update max_iter dynamically if supported
#     if hasattr(model.named_steps[model_name.lower()], 'max_iter'):
#         base_model = model.named_steps[model_name.lower()]
#         if getattr(base_model, 'max_iter', None) and base_model.max_iter < 5000:
#             base_model.set_params(max_iter=5000)

#     if param_grid:
#         search = GridSearchCV(model, param_grid, cv=2, scoring=spearman_scorer, n_jobs=-1, verbose=2)
#         search.fit(X, y)

#         results_df = pd.DataFrame(search.cv_results_)
#         results_df = results_df.sort_values(by="mean_test_score", ascending=False)
        
#         results_df["Model"] = model_name
#         results_df["Antibody"] = antibody
#         results_df["embedding"]  = embedding # make sure `antibody` is passed to the function
#         gridsearch_path = f"{plot_dir}/gridsearch_results_all_models.csv"

#         # Append to file if exists, else write new with header
#         if not os.path.exists(gridsearch_path):
#             results_df.to_csv(gridsearch_path, index=False)
#         else:
#             results_df.to_csv(gridsearch_path, mode='a', header=False, index=False)

#         model = search.best_estimator_
#         best_params = search.best_params_

#     else:
#         if model_name == "GaussianProcessRegressor":
#             try:
#                 model.fit(X, y)
#                 best_params = model.named_steps[model_name.lower()].kernel_
#             except AttributeError:
#                 best_params = "Could not retrieve kernel after training."

#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     mse_scores, pearson_scores, spearman_scores, r2_scores = [], [], [], []
#     under_most_inaccurate, over_most_inaccurate = [], []

#     for train_index, val_index in kf.split(X):
#         X_train, X_val = X[train_index], X[val_index]
#         y_train, y_val = y[train_index], y[val_index]
#         mutations_val = mutations.iloc[val_index] if mutations is not None else None

#         model_fold = clone(model)
#         model_fold.fit(X_train, y_train)

#         y_pred = model_fold.predict(X_val)

#         mse_scores.append(mean_squared_error(y_val, y_pred))
#         pearson_scores.append(pearsonr(y_val, y_pred)[0])
#         spearman_scores.append(spearmanr(y_val, y_pred).correlation)
#         r2_scores.append(r2_score(y_val, y_pred))

#         if mutations_val is not None:
#             under_errors = y_val - y_pred
#             over_errors = -under_errors
#             under_max_error_indices = np.argsort(under_errors)[-3:]
#             over_max_error_indices = np.argsort(over_errors)[-3:]
#             under_most_inaccurate.append(mutations_val.iloc[under_max_error_indices].tolist())
#             over_most_inaccurate.append(mutations_val.iloc[over_max_error_indices].tolist())

#     return {
#         "MSE": (np.mean(mse_scores), np.std(mse_scores)),
#         "Pearson Correlation": (np.mean(pearson_scores), np.std(pearson_scores)),
#         "Spearman Correlation": (np.mean(spearman_scores), np.std(spearman_scores)),
#         "R2": (np.mean(r2_scores), np.std(r2_scores)),
#         "Under Most Inaccurate": under_most_inaccurate,
#         "Over Most Inaccurate": over_most_inaccurate,
#         "Best Hyperparameters": best_params
#     }

def reverse_translate(aa_seq):
    standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
    codon_map = {
        aa: codons[0] for aa, codons in standard_table.forward_table.items()
    }
    codon_map['*'] = 'TAA'  # Stop codon fallback
    return ''.join([codon_map.get(aa, 'NNN') for aa in aa_seq])

class ModelEvaluator:
    def __init__(self, X, y, model, model_name, plot_dir, antibody, embedding, param_grid=None, poly=None, mutations=None):
        self.X = X
        self.y = y
        self.model = model
        self.plot_dir = plot_dir
        self.antibody = antibody
        self.embedding = embedding
        self.param_grid = param_grid
        self.poly = poly
        self.mutations = mutations
        self.model_name = model_name
        self.best_params = []
        self.model_class_name = model.__class__.__name__

        if not isinstance(self.model, Pipeline):
            steps = []
            if self.poly is not None:
                steps.append(('poly', self.poly))
            
            #if self.embedding != "onehot":
            #    steps.append(('standardscaler', StandardScaler()))
            
            steps.append((self.model_class_name.lower(), self.model))
            self.model = Pipeline(steps)

        if hasattr(self.model.named_steps[self.model_class_name.lower()], 'max_iter'):
            base_model = self.model.named_steps[self.model_class_name.lower()]
            if getattr(base_model, 'max_iter', None) and base_model.max_iter < 5000:
                base_model.set_params(max_iter=5000)

    def spearman_corr(self, y_true, y_pred):
        return spearmanr(y_true, y_pred).correlation

    def evaluate(self):
        spearman_scorer = make_scorer(self.spearman_corr, greater_is_better=True)

        if self.param_grid:
            cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
            search = GridSearchCV(self.model, self.param_grid, cv=cv, scoring=spearman_scorer, n_jobs=-1, verbose=2)

            search.fit(self.X, self.y)
            results_df = pd.DataFrame(search.cv_results_)
            results_df = results_df.sort_values(by="mean_test_score", ascending=False)
            results_df["Model"] = self.model_name
            results_df["Antibody"] = self.antibody
            results_df["embedding"] = self.embedding
            gridsearch_path = f"{self.plot_dir}/gridsearch_results_all_models.csv"

            if not os.path.exists(gridsearch_path):
                results_df.to_csv(gridsearch_path, index=False)
            else:
                results_df.to_csv(gridsearch_path, mode='a', header=False, index=False)

            self.model = search.best_estimator_
            self.best_params = search.best_params_
        else:
            if self.model_class_name == "GaussianProcessRegressor":
                try:
                    self.model.fit(self.X, self.y)
                    self.best_params = self.model.named_steps[self.model_class_name.lower()].kernel_
                except AttributeError:
                    self.best_params = "Could not retrieve kernel after training."

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_scores, pearson_scores, spearman_scores, r2_scores = [], [], [], []
        under_most_inaccurate, over_most_inaccurate = [], []

        for train_index, val_index in kf.split(self.X):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]
            mutations_val = self.mutations.iloc[val_index] if self.mutations is not None else None

            model_fold = clone(self.model)
            model_fold.fit(X_train, y_train)
            y_pred = model_fold.predict(X_val)

            mse_scores.append(mean_squared_error(y_val, y_pred))
            pearson_scores.append(pearsonr(y_val, y_pred)[0])
            spearman_scores.append(spearmanr(y_val, y_pred).correlation)
            r2_scores.append(r2_score(y_val, y_pred))

            if mutations_val is not None:
                under_errors = y_val - y_pred
                over_errors = -under_errors
                under_max_error_indices = np.argsort(under_errors)[-3:]
                over_max_error_indices = np.argsort(over_errors)[-3:]
                under_most_inaccurate.append(mutations_val.iloc[under_max_error_indices].tolist())
                over_most_inaccurate.append(mutations_val.iloc[over_max_error_indices].tolist())

        metrics = {
            "Model": self.model_name,
            "Antibody": self.antibody,
            "Embedding": self.embedding,
            "MSE_mean": np.mean(mse_scores),
            "MSE_std": np.std(mse_scores),
            "Pearson_mean": np.mean(pearson_scores),
            "Pearson_std": np.std(pearson_scores),
            "Spearman_mean": np.mean(spearman_scores),
            "Spearman_std": np.std(spearman_scores),
            "R2_mean": np.mean(r2_scores),
            "R2_std": np.std(r2_scores),
            "Under Most Inaccurate": under_most_inaccurate,
            "Over Most Inaccurate": over_most_inaccurate,
            "Best Hyperparameters": self.best_params,
        }

        return metrics
