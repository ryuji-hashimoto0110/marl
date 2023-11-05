seed_max=1
num_agents=3
global_obs_type="AS"
num_nearest_agents=3
local_ratio=0.5
max_cycles=150
algo_name="mappo"
hidden_size=64
rollout_length=2048
num_updates_per_rollout=10
batch_size=1024
gamma=0.995
lr_actor=0.0005
lr_critic=0.001
clip_eps=0.2
lmd=0.97
max_grad_norm=0.5
actor_best_save_name="mappo_spread_best"
actor_last_save_name="mappo_spread_best"
num_train_steps=10000000
eval_interval=100000
num_eval_episodes=10
for seed in `seq ${seed_max}`;
do
    python train_spread.py --num_agents  ${num_agents} --global_obs_type ${global_obs_type} \
    --num_nearest_agents ${num_nearest_agents} --local_ratio ${local_ratio} --max_cycles ${max_cycles} \
    --algo_name ${algo_name} --hidden_size ${hidden_size} --rollout_length ${rollout_length} \
    --num_updates_per_rollout ${num_updates_per_rollout} --batch_size ${batch_size} \
    --gamma ${gamma} --lr_actor ${lr_actor} --lr_critic ${lr_critic} \
    --clip_eps ${clip_eps} --lmd ${lmd} --max_grad_norm ${max_grad_norm} --seed ${seed} \
    --actor_best_save_name ${actor_best_save_name}${seed} \
    --actor_last_save_name ${actor_last_save_name}${seed} \
    --num_train_steps ${num_train_steps} --eval_interval ${eval_interval} \
    --num_eval_episodes ${num_eval_episodes}
done