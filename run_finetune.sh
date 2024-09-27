#!/bin/bash
export WANDB_API_KEY=
#huggingface-cli login --token $HUGGINGFACE_TOKEN

# Hyperparameter ranges
max_learning_rate=1e-4
min_learning_rate=1e-4
max_weight_decay=0.1
min_weight_decay=0.1
max_dropout=0.0
min_dropout=0.0
max_epochs=1
min_epochs=1
grad_accum_steps_values=(16)  # Discrete values for gradient accumulation steps
num_gpus=4
steps_per_epoch=750
per_gpu_batch_size=4
precision="bf16"

# Number of experiments to run
num_experiments=1

for (( i=0; i<num_experiments; i++ )); do
  # Randomly sample hyperparameters
  lr=$(python -c "print($min_learning_rate + ($max_learning_rate - $min_learning_rate) * $RANDOM / 32767)")
  wd=$(python -c "print($min_weight_decay + ($max_weight_decay - $min_weight_decay) * $RANDOM / 32767)")
  dropout=$(python -c "print($min_dropout + ($max_dropout - $min_dropout) * $RANDOM / 32767)")
  epochs=$(( (RANDOM % (max_epochs - min_epochs + 1)) + min_epochs ))
  ga_steps=${grad_accum_steps_values[$RANDOM % ${#grad_accum_steps_values[@]}]}

  # Calculate derived hyperparameters
  global_batch_size=$((num_gpus * ga_steps * per_gpu_batch_size))
  effective_steps_per_epoch=$((steps_per_epoch / ga_steps))
  t_initial=$((epochs * effective_steps_per_epoch))
  warmup_t=$((t_initial / 10))

  echo "Experiment $((i+1)): lr=$lr, wd=$wd, dropout=$dropout, epochs=$epochs, ga_steps=$ga_steps, gbs=$global_batch_size, t_initial=$t_initial, warmup_t=$warmup_t"
  
  # Run training script
  python -m train \
    experiment=hg38/synapse_classification\
    model.d_model=1024 \
    model.n_layer=24 \
    model.checkpoint_mlp=False \
    dataset.batch_size=$per_gpu_batch_size \
    train.global_batch_size=$global_batch_size \
    dataset.max_length=8192 \
    dataset.upsample=False \
    optimizer.lr=$lr \
    optimizer.weight_decay=$wd \
    model.resid_dropout=$dropout \
    model.embed_dropout=0.1 \
    trainer.devices=$num_gpus \
    scheduler.t_initial=$t_initial \
    scheduler.warmup_t=$warmup_t \
    trainer.precision=$precision \
    trainer.max_epochs=$epochs \
    dataset.use_padding=True \
    decoder.mode=last \
    train.pretrained_model_path=/workspace/hyena-rna/checkpoints/Weights/Pretraining/last.ckpt \
    dataset.csv_path=/workspace/hyena-rna/data/mrna/Finetuning_Data/cortex_mouse/cortex_combined.csv \
    dataset.train_fasta_path=/workspace/hyena-rna/data/mrna/Finetuning_Data/mouse_fasta_split_by_gene/P2_cortex_by_gene_filtered_nonvalidation.fasta \
    dataset.validation_fasta_path=/workspace/hyena-rna/data/mrna/Finetuning_Data/mouse_fasta_split_by_gene/P2_cortex_finetuning_by_gene_validation.fasta \
    dataset.test_fasta_path=/workspace/hyena-rna/data/mrna/Finetuning_Data/mouse_fasta_split_by_gene/P2_cortex_finetuning_by_gene_test.fasta \
    dataset.synapse_label_name=synapse_intermediate
done

#cd $HF_FOLDER_PATH
#git add .
#git commit -m "Upload hf_checkpoints folder"
#git push
