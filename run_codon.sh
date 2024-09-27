#!/bin/bash
export WANDB_API_KEY=f32a7be8c3345c866b26e490afad47e206585b50
export HUGGINGFACE_TOKEN=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
#huggingface-cli login --token $HUGGINGFACE_TOKEN

# Hyperparameter ranges
num_gpus=1
per_gpu_batch_size=12
precision="bf16"


python -m train \
    experiment=hg38/codon_scoring \
    model.d_model=1024 \
    model.n_layer=24 \
    model.checkpoint_mlp=False \
    dataset.batch_size=$per_gpu_batch_size \
    dataset.max_length=8192 \
    dataset.upsample=False \
    trainer.devices=$num_gpus \
    trainer.precision=$precision \
    dataset.use_padding=True \
    decoder.mode=pool \
    train.pretrained_model_path=/workspace/hyena-rna/checkpoints/Weights/Finetuning/mouse_cortex_finetuning/threshold_1/mode_pool/last.ckpt \
    dataset.predict_fasta_path=/workspace/hyena-rna/data/mrna/Prediction_Data/Codon_Data/mutated_hotspot_sequences_seed_3_all_uppercase.fasta \
    train.predict=True 


#cd $HF_FOLDER_PATH
#git add .
#git commit -m "Upload hf_checkpoints folder"
#git push