#!/bin/bash

# Define the arguments
ds_name="ARC-Challenge"
model_name="Qwen2"
seeds=(0 1 2)

# Loop through n_shots and seeds
for n_shots in {0..5}; do
  for seed in "${seeds[@]}"; do
    python main.py --ds_name $ds_name --model_name $model_name --n_shots $n_shots --seed $seed
  done
done
