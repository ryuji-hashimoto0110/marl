import argparse

def get_config_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="mappo",
                        choices=["mappo"])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--rollout_length", type=int, default=3072)
    parser.add_argument("--num_updates_per_rollout", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr_actor", type=float, default=5e-04)
    parser.add_argument("--lr_critic", type=float, default=1e-03)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--lmd", type=float, default=0.97)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor_best_save_name", type=str, required=False)
    parser.add_argument("--actor_last_save_name", type=str, required=False)
    parser.add_argument("--num_train_steps", type=int, default=int(1e+07))
    parser.add_argument("--eval_interval", type=int, default=int(1e+05))
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    return parser

def get_config_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_name", type=str, default="mappo",
                        choices=["mappo"])
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--actor_load_name", type=str, required=False)
    parser.add_argument("--video_name", type=str, required=False)
    parser.add_argument("--video_width", type=int, default=256)
    parser.add_argument("--video_height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=256)
    parser.add_argument("--reward_font_size", type=int, default=20)
    return parser