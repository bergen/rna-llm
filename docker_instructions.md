export WANDB_API_KEY=f32a7be8c3345c866b26e490afad47e206585b50

#for atlantic:
sudo docker run --privileged --gpus all --rm -e WANDB_API_KEY=${WANDB_API_KEY} \
-v /home/lbergen/hyena-rna:/workspace \
-v /data/lbergen/mrna/finetuning:/workspace/data/mrna/finetuning \
-w /workspace -it a8985b55b539

#for oblivus:
sudo docker run --privileged --gpus all --rm -e WANDB_API_KEY=${WANDB_API_KEY} \
-v /home/user/hyena-rna:/workspace \
-w /workspace -it a8985b55b539


Things to do when starting new server:
1. setup github 
2. create docker image
3. transfer pretraining/finetuning data to data/mrna/finetuning or data/mrna/pretraining
4. change image name in docker commands
5. update num gpus in run_finetune.sh and run.sh

