seed_max=1
num_agents=3
global_obs_type="AS"
num_nearest_agents=2
num_nearest_landmarks=3
local_ratio=0.5
max_cycles=25
num_episodes=5
algo_name="mappo"
hidden_size=128
actor_load_name="mappo_spread_best1"
video_width=256
video_height=256
fps=15
reward_font_size=15

for seed in `seq ${seed_max}`;
do
    python evaluate_spread.py --num_agents  ${num_agents} --global_obs_type ${global_obs_type} \
    --num_nearest_agents ${num_nearest_agents} --num_nearest_landmarks ${num_nearest_landmarks} \
    --local_ratio ${local_ratio} --max_cycles ${max_cycles} \
    --algo_name ${algo_name} --hidden_size ${hidden_size} --seed ${seed} \
    --actor_load_name ${actor_load_name} --num_episodes ${num_episodes} \
    --video_name spread_${algo_name}_${global_obs_type}_${num_agents}-${num_nearest_agents}agents \
    --video_width ${video_width} --video_height ${video_height} --fps ${fps} \
    --reward_font_size ${reward_font_size}
done