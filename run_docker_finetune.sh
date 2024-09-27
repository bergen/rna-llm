export WANDB_API_KEY=
docker run --privileged --gpus all --rm \
-e WANDB_API_KEY=${WANDB_API_KEY} \
-v /home/user/hyena-rna:/workspace \
-w /workspace a8985b55b539 /bin/bash run_finetune.sh
