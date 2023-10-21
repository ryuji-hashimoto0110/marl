from algorithm import Algorithm
import argparse
from gym import Env
import numpy as np
from numpy import ndarray
from pathlib import Path
from pettingzoo.utils.env import ParallelEnv
from typing import Any, Optional, TypeVar

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class Trainer:
    """Trainer class.

    Train and evaluate RL algorithm.
    """
    def __init__(
        self,
        train_env: Env | ParallelEnv,
        test_env: Env | ParallelEnv,
        algo: Algorithm,
        seed: int = 1111,
        actor_best_save_path: Optional[Path] = None,
        actor_last_save_path: Optional[Path] = None,
        critic_best_save_path: Optional[Path] = None,
        critic_last_save_path: Optional[Path] = None,
        other_indicators: list[str] = [],
        num_train_steps: int = int(1e+07),
        eval_interval: int = int(1e+05),
        num_eval_episodes: int = 10
    ) -> None:
        """initialization

        Args:
            train_env (Env | ParallelEnv): _description_
            test_env (Env | ParallelEnv): _description_
            algo (Algorithm): _description_
            seed (int, optional): _description_. Defaults to 1111.
            actor_best_save_path (Path, optional): _description_. Defaults to None.
            actor_last_save_path (Path, optional): _description_. Defaults to None.
            critic_best_save_path (Path, optional): _description_. Defaults to None.
            critic_last_save_path (Path, optional): _description_. Defaults to None.
            other_indicators (list[str]): _description_. Defaults to [],
            num_train_steps (int, optional): _description_. Defaults to int(1e+07).
            eval_interval (int, optional): _description_. Defaults to int(1e+05).
            num_eval_episodes (int, optional): _description_. Defaults to 10.
        """
        self.train_env: Env | ParallelEnv = train_env
        self.train_env.seed(seed)
        self.test_env: Env | ParallelEnv = test_env
        seed += 1111
        self.test_env.seed(seed)
        self.algo: Algorithm = algo
        self.actor_best_save_path: Optional[Path] = actor_best_save_path
        self.actor_last_save_path: Optional[Path] = actor_last_save_path
        self.critic_best_save_path: Optional[Path] = critic_best_save_path
        self.critic_last_save_path: Optional[Path] = critic_last_save_path
        self.other_indicators: list[str] = other_indicators
        self.results_dic: dict[str, list[float]] = self.set_results_dic(self.other_indicators)
        self.num_train_steps: int = int(num_train_steps)
        self.eval_interval: int = int(eval_interval)
        self.num_eval_episodes: int = int(num_eval_episodes)
        self.best_reward: float = - 1e-10

    def set_results_dic(self, other_indicators: list[str]) -> dict[str, list[float]]:
        """set initial results dictionary.

        Initialize results_dic.
        By default, only total_reward is set as the indicator.
        Extend results_dic by adding other_indicators.

        Args:
            other_indicators (list[str]): _description_

        Returns:
            results_dic (dict[str, list[float]]): _description_
        """
        results_dic = dict[str, list[float]] = {"step": [], "total_reward": []}
        for indicator in other_indicators:
            results_dic[indicator]: list[float] = []
        return results_dic

    def train(self) -> None:
        """train algorithm.

        Train self.algo for num_train_steps steps in self.train_env environment.
        Conduct evaluations for num_eval_episodes episodes in self.test_env environment
            for each of eval_interval training steps.
        """
        current_episode_steps: int = 0
        obs: ObsType | tuple[dict[AgentID, ObsType]] = self.train_env.reset()
        for current_total_steps in range(1, self.num_train_steps+1):
            obs, current_episode_steps = \
                self.algo.step(self.train_env, obs, current_episode_steps, current_total_steps)
            if self.algo.is_ready_to_update(current_total_steps):
                self.algo.update()
            if current_total_steps % self.eval_interval == 0:
                self.evaluate(current_total_steps)

    def evaluate(self, current_total_steps: int) -> None:
        """evaluate algorithm.

        Run episodes self.num_eval_episodes times in self.test_env environment.
        Record average cumulative rewards and other indicators.
        Save paremeters of the trained network.

        Args:
            current_total_steps (int): _description_
        """
        eval_rewards: list = []
        for _ in range(self.num_eval_episodes):
            obs: ObsType | tuple[dict[AgentID, ObsType]] = self.test_env.reset()
            done: bool = False
            episode_reward: float = 0.0
            while not done:
                action: ActionType | dict[AgentID, ActionType] = self.algo.exploit(obs)
                obs, reward, done, info = self.step_env(self.test_env, action)
                episode_reward += reward
            eval_rewards.append(episode_reward)
        eval_average_reward: float = np.mean(eval_rewards)
        self.record_indicators(current_total_steps, eval_average_reward, info)
        if self.actor_best_save_path is not None:
            self.save_params(eval_average_reward)

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
            reward: float = sum(rewards.values) / len(rewards.values)
            done: bool = False
            if sum(terminations.values()) > 0 or sum(truncations.values()) > 0:
                done = True
        elif isinstance(env, Env):
            obs, reward, done, info = env.step(action)
        return obs, reward, done, info

    def record_indicators(
        self,
        current_total_steps: int,
        eval_average_reward: float,
        info: dict[str, Any] | dict[AgentID, dict[str, Any]]
    ) -> None:
        """_summary_

        Args:
            current_total_steps (int): _description_
            eval_average_reward (float): _description_
            info (dict[str, Any] | dict[AgentID, dict[str, Any]]): _description_
        """
        self.results_dic["step"].append(current_total_steps)
        self.results_dic["total_reward"].append(eval_average_reward)

    def save_params(self, eval_average_reward: float) -> None:
        """_summary_

        Args:
            eval_average_reward (float): _description_
        """
        # parameter sharingの場合もあるから個別に実装する．
        self.algo.save_params(self.best_reward, eval_average_reward)