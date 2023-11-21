import argparse
from config import get_config_eval
import numpy as np
from numpy import ndarray
import pathlib
from pathlib import Path
from pettingzoo import ParallelEnv
import sys
import torch
from typing import Optional
curr_path: Path = pathlib.Path(__file__).resolve()
grandparent_path: Path = curr_path.parents[1] # ~/marl
sys.path.append(str(grandparent_path))
from drl_algos import Algorithm
from drl_algos import MAPPO
from drl_algos import Evaluater
from environments.simple_spread import make_ep_spread_env

def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--global_obs_type", type=str, default="EP", choices=["EP", "AS"])
    parser.add_argument("--num_nearest_agents", type=int, default=2,
                        help="number of nearest agents that can be observed by each agent.")
    parser.add_argument("--num_nearest_landmarks", type=int, default=3,
                        help="number of nearest landmarks that can be observed by each agent.")
    parser.add_argument("--local_ratio", type=float, default=0.5,
                        help="ratio of local reward to global reward.")
    parser.add_argument("--max_cycles", type=int, default=25,
                        help="number of frames (a step for each agent) until game terminates.")
    parser.add_argument("--num_episodes", type=int, default=150)
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config_eval()
    all_args = parse_args(args, parser)
    num_agents: int = all_args.num_agents
    global_obs_type: str = all_args.global_obs_type
    num_nearest_agents: str = all_args.num_nearest_agents
    num_nearest_landmarks: int = all_args.num_nearest_landmarks
    local_ratio: float = all_args.local_ratio
    max_cycles: int = all_args.max_cycles
    num_episodes: int = all_args.num_episodes
    print("===Simple Spread Environment.===")
    env: ParallelEnv = make_ep_spread_env(
        num_agents, num_nearest_agents, num_nearest_landmarks, local_ratio, max_cycles
    )
    obs_shape: tuple[int] = env.observation_space("agent_0").shape
    if global_obs_type == "EP":
        global_obs_shape: tuple[int] = env.global_obs_shape.shape
    elif global_obs_type == "AS":
        global_obs_shape: tuple[int] = tuple(np.add(
            env.global_obs_shape.shape, obs_shape
        ))
    action_shape: tuple[int] = env.action_space("agent_0").shape
    print(f"number of agents={num_agents}, number of nearest agents and landmarks that can be observed={num_nearest_agents}")
    print(f"local ratio={local_ratio}, max cycles={max_cycles} number of episodes={num_episodes}")
    print(f"obs space shape={obs_shape}, global obs space shape={global_obs_shape}, action space shape={action_shape}")
    print()
    algo_name: str = all_args.algo_name
    if algo_name == "mappo":
        print("You are choosing to use MAPPO with reccurent policy.")
        print(f"you are choosing to use {global_obs_type} as global observation type.")
        hidden_size: int = all_args.hidden_size
        agent_ids: list[str] = [f"agent_{id}" for id in range(num_agents)]
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        algo: Algorithm = MAPPO(
            obs_shape=obs_shape,
            global_obs_shape=global_obs_shape,
            global_obs_type=global_obs_type,
            action_shape=action_shape,
            hidden_size=hidden_size,
            agent_ids=agent_ids,
            device=device
        )
        print(f"device={device}")
        print(f"hidden state size of policy GRU={hidden_size}")
        print()
    seed: int = all_args.seed
    actor_load_name: Optional[str] = all_args.actor_load_name
    actor_load_path: Optional[Path] = None
    checkpoints_path: Path = grandparent_path / "checkpoints"
    if actor_load_name is not None:
        actor_load_name = actor_load_name + ".pth"
        actor_load_path = checkpoints_path / actor_load_name
    evaluater = Evaluater(
        algo, num_episodes, seed, actor_load_path, env, device
    )
    video_name: Optional[str] = all_args.video_name
    videos_path: Path = grandparent_path / "rendered_video"
    if video_name is not None:
        video_name = video_name + ".mp4"
        video_save_path = videos_path / video_name
        video_width: int = all_args.video_width
        video_height: int = all_args.video_height
        fps: int = all_args.fps
        reward_font_size: int = all_args.reward_font_size
        evaluater.make_video(
            video_save_path, video_width, video_height, fps, reward_font_size
        )

if __name__ == "__main__":
    main(sys.argv[1:])