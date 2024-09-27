export WANDB_API_KEY=f32a7be8c3345c866b26e490afad47e206585b50
docker run --privileged --gpus all --rm \
-e WANDB_API_KEY=${WANDB_API_KEY} \
-v /home/user/hyena-rna:/workspace \
-w /workspace a8985b55b539 /bin/bash run.sh