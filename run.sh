TORCHRUN="torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=1"                                                             
cmd="${TORCHRUN} main.py --debug 1 --use_wandb 1 --dataset cifar --process_name rf_tied"
echo ${cmd}
eval ${cmd}
