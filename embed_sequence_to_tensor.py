import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from utils import one_hot_encode_sequences, pool_embeddings
from datetime import datetime
import torch
#import esm
from transformers import AutoTokenizer, EsmModel
#from antiberty import AntiBERTyRunner
import argparse
#from align import align_sequences

# Current time
now = datetime.now()
print(now)

# Argument parser
parser = argparse.ArgumentParser(description='embed amino acid sequence and save as tensor files in scratch folder.')

# Adding arguments
parser.add_argument('--embedding', type=str, default='esm1v', 
                    choices=['onehot', 'antiberty', 'esm2', 'esm1v'], 
                    help='Type of embedding to use.')
parser.add_argument('--esm_layer', type=int, default=15,
                    help='Embedding layer in ESM2.')
parser.add_argument('--pool_method', type=str, default='mean',
                    help='Pooling method for PLM embeddings.')
parser.add_argument('--pool_axis', type=int, default=0,
                    help='Pooling axis for embeddings.')        
parser.add_argument('--y_name', type=str, default='Mean_Relative_Affinity',
                    help='End point column name.')
parser.add_argument('--task_name', type=str, default='moulana',
                    help='Task name, same as folder name inside data folder, script name prefix for submission sh file and python file.')                    
# Parse the arguments
args = parser.parse_args() 

embedding = args.embedding
esm_layer = args.esm_layer
pool_method = args.pool_method
pool_axis = args.pool_axis
y_name = args.y_name
task_name = args.task_name

argstr = f"{task_name}_{embedding}_layer{esm_layer}_{pool_method}pool{pool_axis}"
#data_path = 'data/data_EPFL.csv'
data_path = "/cluster/home/jiahan/SSHPythonProject/antibody_kd_regression/data/RBD_validation_set.csv"
df = pd.read_csv(data_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_sequence(seq):
    return seq[330:531]

#cut RBD from the spike sequence.
#sequences = df['spike_aa'].apply(adjust_sequence) 
sequences = df['spike_aa']

if embedding == 'onehot':
    #aligned_sequences = align_sequences(df['spike_aa'].tolist())
    #aligned_sequences = sequences
    #df['spike_aa'] = aligned_sequences
    #df.to_csv('data_EPFL_aligned.csv')
    #print("Saved aligned CSV")
    X = one_hot_encode_sequences(sequences, aa_list='ACDEFGHIKLMNPQRSTVWYZX-')
    X = [x.flatten() for x in X]
    X = np.stack(X)  # This results in a numpy array, which is fine for one-hot
elif embedding == 'antiberty':
    model_name = "Rostlab/AntiBERTy"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def batch_process_antiberty(sequences, batch_size):
        batched_embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Tokenize the batch of sequences
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                # Get the model output (last hidden state or specific hidden layer)
                outputs = model(**inputs, output_hidden_states=True)
            
            # Extract the last hidden state
            last_hidden_state = outputs.last_hidden_state
            
            # Process each sequence individually (if batch size > 1)
            for seq_idx in range(last_hidden_state.size(0)):
                embedding = last_hidden_state[seq_idx].cpu()
                batched_embeddings.append(embedding)
            
            # Clear cache to free up memory
            torch.cuda.empty_cache()
        return batched_embeddings

    batch_size = 8  # Adjust batch size as needed
    X = batch_process_antiberty(sequences, batch_size)
    #X = [pool_embeddings(x, pool_method, pool_axis) for x in X]
    X = torch.stack(X)
    X = X.cpu()  # Use torch.stack to concatenate tensors
elif embedding == 'esm2':
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model.eval()
    model = model.to(device)

    batch_size = 8
    sequence_representations = []

    for start_idx in range(0, len(sequences), batch_size):
        end_idx = start_idx + batch_size
        data_batch = sequences[start_idx:end_idx].tolist()
    
    # Tokenize the sequences
        inputs = tokenizer(data_batch, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            # Forward pass through the model
            results = model(**inputs.to(device), output_hidden_states=True)
    
        # Retrieve the hidden representations from the desired layer
        hidden_states = results.hidden_states[esm_layer]

        for i, tokens_len in enumerate((inputs['input_ids'] != tokenizer.pad_token_id).sum(1)):
            # Extract representations for the sequence (excluding padding)
            seq_rep = hidden_states[i, 1:tokens_len - 1].cpu()
            if pool_method == 'unpool':
                sequence_representations.append(seq_rep)
            else:
                pooled_rep = pool_embeddings(seq_rep, pool_method)
                sequence_representations.append(pooled_rep)


    sequence_representations_tensors = [torch.tensor(rep, dtype=torch.float) for rep in sequence_representations]
    X = torch.stack(sequence_representations_tensors)

elif embedding == 'esm1v':
    # Load ESM1v model and alphabet
    model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)

    batch_size = 8
    sequence_representations = []

    for start_idx in range(0, len(sequences), batch_size):
        end_idx = start_idx + batch_size
        data_batch = [(f"protein_{i}", seq) for i, seq in enumerate(sequences[start_idx:end_idx])]
        batch_labels, batch_strs, batch_tokens = batch_converter(data_batch)

        with torch.no_grad():
            results = model(batch_tokens.to(device), repr_layers=[esm_layer])

        token_representations = results["representations"][esm_layer].cpu()
        for i, tokens_len in enumerate((batch_tokens != alphabet.padding_idx).sum(1)):
            seq_rep = token_representations[i, 1:tokens_len - 1]
            pooled_rep = pool_embeddings(seq_rep, pool_method)
            sequence_representations.append(pooled_rep)

    sequence_representations_tensors = [torch.tensor(rep, dtype=torch.float) for rep in sequence_representations]
    X = torch.stack(sequence_representations_tensors)

print(X.shape)
torch.save(X, f'/cluster/project/reddy/jiahan/{argstr}_tensor.pt')
print(f'Saved tensor shape {X.shape}')
