#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100g
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --job-name=gp_seed0



module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load python/3.11.6 cuda/12.1.1 ninja/1.11.1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/cluster/project/reddy/jiahan/my_venv/lib/python3.11/site-packages:$PYTHONPATH"
source /cluster/project/reddy/jiahan/my_venv/bin/activate

pool=('mean')
embedding=('16encode' 'esm2' 'onehot')

for len in 1
do
    for layer in 15
    do
        for pl in "${pool[@]}"
        do
            for pa in 0 
            do 
                for emb in "${embedding[@]}"
                do
                    #python /cluster/home/jiahan/SSHPythonProject/antibody_kd_regression/esm1v_embedding.py --embedding "$emb" --esm_layer $layer --pool_method "$pl" --pool_axis $pa --task_name valexp
                    #python /cluster/home/jiahan/SSHPythonProject/antibody_kd_regression/esm1v_embedding.py --embedding "$emb" --esm_layer $layer --pool_method "$pl" --pool_axis $pa --task_name valsetRBD
                    #python /cluster/home/jiahan/SSHPythonProject/antibody_kd_regression/readtensor.py --embedding "$emb" --mlp_hidden_dim 64 --length_scale $len --embedding "$emb" --esm_layer $layer --pool_method "$pl" --pool_axis $pa --task_name EPFL
                    python /cluster/home/jiahan/SSHPythonProject/antibody_kd_regression/readtensor_cv.py --embedding "$emb" --mlp_hidden_dim 64 --length_scale $len --embedding "$emb" --esm_layer $layer --pool_method "$pl" --pool_axis $pa --task_name EPFL
                done
            done
        done
    done
done