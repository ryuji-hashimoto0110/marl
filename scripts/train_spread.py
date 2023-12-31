import argparse
from config import get_config_train
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
from drl_algos import Trainer
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
    parser.add_argument("--max_cycles", type=int, default=150,
                        help="number of frames (a step for each agent) until game terminates.")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config_train()
    all_args = parse_args(args, parser)
    num_agents: int = all_args.num_agents
    global_obs_type: str = all_args.global_obs_type
    num_nearest_agents: int = all_args.num_nearest_agents
    num_nearest_landmarks: int = all_args.num_nearest_landmarks
    local_ratio: float = all_args.local_ratio
    max_cycles: int = all_args.max_cycles
    print("===Simple Spread Environment.===")
    train_env: ParallelEnv = make_ep_spread_env(
        num_agents, num_nearest_agents, num_nearest_landmarks, local_ratio, max_cycles
    )
    valid_env: ParallelEnv = make_ep_spread_env(
        num_agents, num_nearest_agents, num_nearest_landmarks, local_ratio, max_cycles
    )
    obs_shape: tuple[int] = train_env.observation_space("agent_0").shape
    if global_obs_type == "EP":
        global_obs_shape: tuple[int] = train_env.global_obs_shape.shape
    elif global_obs_type == "AS":
        global_obs_shape: tuple[int] = tuple(np.add(
            train_env.global_obs_shape.shape, obs_shape
        ))
    action_shape: tuple[int] = train_env.action_space("agent_0").shape
    print(f"number of agents={num_agents}, number of nearest agents and landmarks that can be observed={num_nearest_agents}")
    print(f"local ratio={local_ratio}, max cycles={max_cycles}")
    print(f"obs space shape={obs_shape}, global obs space shape={global_obs_shape}, action space shape={action_shape}")
    print()
    algo_name: str = all_args.algo_name
    if algo_name == "mappo":
        print("You are choosing to use MAPPO with reccurent policy.")
        print(f"you are choosing to use {global_obs_type} as global observation type.")
        hidden_size: int = all_args.hidden_size
        rollout_length: int = all_args.rollout_length
        num_updates_per_rollout: int = all_args.num_updates_per_rollout
        batch_size: int = all_args.batch_size
        gamma: float = all_args.gamma
        lr_actor: float = all_args.lr_actor
        lr_critic: float = all_args.lr_critic
        clip_eps: float = all_args.clip_eps
        lmd: float = all_args.lmd
        max_grad_norm: float = all_args.max_grad_norm
        agent_ids: list[str] = [f"agent_{id}" for id in range(num_agents)]
        device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        algo: Algorithm = MAPPO(
            obs_shape=obs_shape,
            global_obs_shape=global_obs_shape,
            global_obs_type=global_obs_type,
            action_shape=action_shape,
            hidden_size=hidden_size,
            agent_ids=agent_ids,
            device=device,
            rollout_length=rollout_length,
            num_updates_per_rollout=num_updates_per_rollout,
            batch_size=batch_size,
            gamma=gamma,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            clip_eps=clip_eps,
            lmd=lmd,
            max_grad_norm=max_grad_norm
        )
        print(f"device={device}")
        print(f"hidden state size of policy GRU={hidden_size}")
        print(f"rollout length={rollout_length}, number of times to update per 1 rollout={num_updates_per_rollout}, batch size={batch_size}")
        print(f"discount rate={gamma}, learning rate actor={lr_actor}, critic={lr_critic}")
        print(f"threshold to clip importance_ratio={clip_eps}, lambda in GAE={lmd}, max gradient norm={max_grad_norm}")
        print()
    seed: int = all_args.seed
    actor_best_save_name: Optional[str] = all_args.actor_best_save_name
    actor_last_save_name: Optional[str] = all_args.actor_last_save_name
    actor_best_save_path: Optional[Path] = None
    actor_last_save_path: Optional[Path] = None
    checkpoints_path: Path = grandparent_path / "checkpoints"
    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)
    if actor_best_save_name is not None:
        actor_best_save_name = actor_best_save_name + ".pth"
        actor_best_save_path = checkpoints_path / actor_best_save_name
    if actor_last_save_name is not None:
        actor_last_save_name = actor_last_save_name + ".pth"
        actor_last_save_path = checkpoints_path / actor_last_save_name
    num_train_steps: int = all_args.num_train_steps
    eval_interval: int = all_args.eval_interval
    num_eval_episodes: int = all_args.num_eval_episodes
    trainer = Trainer(
        train_env=train_env,
        test_env=valid_env,
        algo=algo,
        seed=seed,
        actor_best_save_path=actor_best_save_path,
        actor_last_save_path=actor_last_save_path,
        other_indicators=[],
        num_train_steps=num_train_steps,
        eval_interval=eval_interval,
        num_eval_episodes=num_eval_episodes
    )
    print(f"seed={seed}")
    print(f"actor best save path={actor_best_save_path}, actor last save path={actor_last_save_path}")
    print(f"number of train steps={num_train_steps}")
    print(f"evaluation interval={eval_interval}")
    print(f"number of evaluation episodes={num_eval_episodes}")
    print()
    trainer.train()

if __name__ == "__main__":
    main(sys.argv[1:])