#!/bin/sh
env="CellFree"
scenario="cell_free" 
num_agents=6
algo="rmappo" #"mappo" "ippo"
exp="MAPPO"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=5 python onpolicy/train_tii.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 --episode_length 10 --num_env_steps 1280000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "xumanba-tsinghua-university"
done
