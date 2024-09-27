#!/bin/bash

export HUGGINGFACE_TOKEN=
#huggingface-cli login --token $HUGGINGFACE_TOKEN

# Get GPU models and total memory
GPU_MODELS=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | sed 's/ MiB//')
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if echo "${GPU_MODELS}" | grep -q "A100"; then
    MEM_VALUE=$(echo "${GPU_MEMORY}" | head -n1)
    if [ "$MEM_VALUE" -eq "40960" ]; then
        echo "A100 GPU (40GB version) detected."
        precision="bf16"
        batch_size="8"
    elif [ "$MEM_VALUE" -eq "81920" ]; then
        echo "A100 GPU (80GB version) detected."
        precision="bf16"
        batch_size="8"
    else
        echo "A100 GPU with unexpected memory size detected: ${MEM_VALUE} MiB"
    fi
elif echo "${GPU_MODELS}" | grep -q "V100"; then
    MEM_VALUE=$(echo "${GPU_MEMORY}" | head -n1)
    if [ "$MEM_VALUE" -eq "16384" ]; then
        echo "V100 GPU (16GB version) detected."
        precision="16"
        batch_size="4"
    elif [ "$MEM_VALUE" -eq "32768" ]; then
        echo "V100 GPU (32GB version) detected."
        precision="16"
        batch_size="8"
    else
        echo "V100 GPU with unexpected memory size detected: ${MEM_VALUE} MiB"
    fi
elif echo "${GPU_MODELS}" | grep -q "A6000"; then
    MEM_VALUE=$(echo "${GPU_MEMORY}" | head -n1)
    echo "A6000 GPU detected."
    precision="bf16"
    batch_size="16" 
elif echo "${GPU_MODELS}" | grep -q "H100"; then
    echo "H100 GPU detected."
    precision="bf16"
    batch_size="1"
else
    echo "No compatible GPU detected."
fi


python -m train \
  experiment=hg38/mrna \
  model.d_model=256 \
  model.n_layer=8 \
  dataset.batch_size=$batch_size \
  train.global_batch_size=112 \
  dataset.max_length=4096 \
  optimizer.lr=2e-4 \
  trainer.devices=$NUM_GPUS \
  scheduler.t_initial=48000 \
  scheduler.warmup_t=2000 \
  trainer.precision=$precision \
  trainer.max_epochs=4 \
  dataset.fasta_directory=/workspace/hyena-rna/data/fake_test


#cd $HF_FOLDER_PATH
#git add .
#git commit -m "Upload hf_checkpoints folder"
#git push
