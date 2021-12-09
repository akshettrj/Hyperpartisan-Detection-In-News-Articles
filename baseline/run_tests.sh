# !/bin/bash

for seed in `seq 1 100`; do
    echo "Running for seed = $seed"
    PYTHONPATH=./elmo/bilm-tf python3 elmo_preprocess.py $seed
done
