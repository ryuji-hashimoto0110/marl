import cv2
from gym import Env
import numpy as np
from numpy import ndarray
import pathlib
from pathlib import Path
import sys
from pettingzoo.utils.env import ParallelEnv
import torch
from typing import Any, Optional, TypeVar
curr_path: pathlib.Path = pathlib.Path(__file__).resolve()
parent_path: pathlib.Path = curr_path.parents[0]
sys.path.append(str(parent_path))
from algorithm import Algorithm

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class Evaluater:
    def __init__(
        self,
        algo: Algorithm,
        num_episodes: int,
        seed: int,
        actor_load_path: Path,
        env: Env | ParallelEnv,
        device: torch.device
    ):
        self.algo: Algorithm = algo
        self.num_episodes: int = num_episodes
        self.env: env = env
        self.env.seed(seed)
        self.device: torch.device = device
        self.actor_load(actor_load_path)

    def actor_load(
        self,
        actor_load_path: Path
    ) -> None:
        self.algo.actor.load_state_dict(
            torch.load(str(actor_load_path), map_location=self.device)["actor_state_dict"]
        )

    @torch.no_grad()
    def make_video(
        self,
        video_save_path: Path,
        width: int = 256,
        height: int = 256,
        fps: int = 15,
        reward_font_size: int = 20
    ) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(video_save_path), fourcc, fps, (width, height))
        for _ in range(self.num_episodes):
            done: bool = False
            obs: ObsType | tuple[dict[AgentID, ObsType], Any] = self.env.reset()
            if isinstance(obs, tuple):
                obs: dict[AgentID, ObsType] = obs[0]
            self.algo.actor.initialize_h()
            while not done:
                action: ActionType | dict[AgentID, ActionType] = self.algo.exploit(obs)
                obs, reward, done, info = self.step_env(self.env, action)
                frame: ndarray = self.env.render()
                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (width, height))
                if reward_font_size > 0:
                    frame = cv2.putText(
                        frame,
                        f"reward:{reward:.2f}",
                        (0, height-reward_font_size),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        (0,0,255),
                        1,
                        cv2.LINE_AA
                    )
                video.write(frame)
        video.release()

    def step_env(
        self, env: Env | ParallelEnv,
        action: ActionType | dict[AgentID, ActionType]
    ) -> tuple[ObsType, float, bool]:
        """_summary_

        Args:
            env (Env | ParallelEnv): _description_
            action (ActionType): _description_

        Returns:
            tuple[ObsType, float, bool]: _description_
        """
        if isinstance(env, ParallelEnv):
            obs, rewards, terminations, truncations, info = env.step(action)
            reward: float = sum(rewards.values()) / len(rewards.values())
            done: bool = False
            if sum(terminations.values()) > 0 or sum(truncations.values()) > 0:
                done = True
        elif isinstance(env, Env):
            obs, reward, done, info = env.step(action)
        return obs, reward, done, info