import cv2
from gym.spaces import Box
from gymnasium.spaces import Space
import numpy as np
from numpy import ndarray
import pathlib
from pathlib import Path
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import BaseParallelWrapper
from pettingzoo.utils.env import ParallelEnv
import torch
from typing import Any, Dict, TypeVar

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

inf = float("inf")

class LimitLocalObsSimpleSpreadEnv(BaseParallelWrapper):
    """limit local observations.
    """
    def __init__(
        self,
        env: ParallelEnv,
        num_nearest_agents: int,
        num_nearest_landmarks: int,
        num_agents: int
    ):
        BaseParallelWrapper.__init__(self, env)
        self.n_agents: int = num_nearest_agents
        self.n_landmarks: int = num_nearest_landmarks
        self.N: int = num_agents
        if self.N < self.n_agents:
            raise ValueError(
                f"num_nearest_agents is greater than num_agent ({self.N}<{self.n_agents})"
            )
        if self.N < self.n_landmarks:
            raise ValueError(
                f"num_nearest_agents is greater than num_agent ({self.N}<{self.n_landmarks})"
            )

    def observation_space(
        self,
        agent: AgentID
    ) -> Space:
        """change observation space.

        Observation space of each agent is Box(-inf, inf, (6*N,), float32)
        in simple spread by default, cinsisting of:
            - physical velocity of the agent. (dim=2)
            - physical position (coordinate) of the agent. (dim=2)
            - relative position of each landmarks from the agent. (dim=2*N)
            - relative position of other agents from the agent. (dim=2*(N-1))
            - communication with other agents. (2*(N-1))
        This wrapper prune the observation to:
            - physical velocity of the agent. (dim=2)
            - physical position (coordinate) of the agent. (dim=2)
            - relative position of nearest n landmarks from the agent. (dim=2*n_landmarks)
            - relative position of nearest n agents from the agent. (dim=2*n_agents)

        Args:
            agent (AgentID): _description_
        """
        return Box(
            -inf, inf,
            shape=(2*(2 + self.n_landmarks + self.n_agents),),
            dtype=np.float32
        )

    def reset(self, **kwargs) -> tuple[dict[AgentID, ObsType], dict]:
        obs_dic, info_dic = super().reset(**kwargs)
        for agent_id, obs in obs_dic.items():
            obs_dic[agent_id] = self.prune_obs(obs)
        return obs_dic, info_dic

    def step(
        self,
        actions: dict[AgentID, ActionType]
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, float],
            dict[AgentID, bool], dict[AgentID, bool],
            dict[AgentID, Any]]:
        next_obs_dic, reward_dic, done_dic, truncation_dic, info_dic = super().step(actions)
        for agent_id, obs in next_obs_dic.items():
            next_obs_dic[agent_id] = self.prune_obs(obs)
        return next_obs_dic, reward_dic, done_dic, truncation_dic, info_dic

    def prune_obs(self, obs: ObsType) -> ObsType:
        """_summary_

        Args:
            obs (ObsType): _description_

        Returns:
            ObsType: _description_
        """
        vel: ObsType = obs[:2]
        pos: ObsType = obs[2:4]
        landmarks_idx: int = 4
        pos_near_landmarks: ObsType = obs[landmarks_idx:landmarks_idx+2*self.n_landmarks]
        agents_idx: int = 4+2*(self.N)
        pos_near_agents: ObsType = obs[agents_idx:agents_idx+2*self.n_agents]
        obs: ObsType = np.concatenate(
            [vel, pos, pos_near_landmarks, pos_near_agents]
        )
        return obs

class ModifyActionEnv(BaseParallelWrapper):
    def step(
        self,
        actions: dict[AgentID, ActionType]
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, float],
            dict[AgentID, bool], dict[AgentID, bool],
            dict[AgentID, Any]]:
        for agent_id, action in actions.items():
            actions[agent_id] = 0.5 + action / 2
        return super().step(actions)

class EPSimpleSpreadEnv(BaseParallelWrapper):
    def __init__(
        self,
        env: ParallelEnv,
        num_agents: int
    ):
        BaseParallelWrapper.__init__(self, env)
        self.N: int = num_agents
        self.global_obs_shape: Space = Box(-inf, inf, shape=(6*self.N,))

    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    def state(self) -> ObsType:
        """generate global state.

        Default simple-spread environment provide global state
        which is a concatenation of local observations of all agents by built-in method .stete().
        This wrapper provide global state that is consisted of:
            - physical velocity of all agents. (dim=2*N)
            - physical position (coordinate) of all agents. (dim=2*N)
            - physical position (coordinate) of all landmarks. (dim=2*N)
        """
        N: int = self.N
        global_obs = self.env.state()
        if len(global_obs) == 0:
            return np.zeros(6*N)
        global_obs = global_obs.reshape(N,N*6)
        all_vel = global_obs[:,:2].reshape(2*N)
        all_pos = global_obs[:,2:4].reshape(2*N)
        landmarks_idx: int = 4
        all_landmark_pos = \
            (
                global_obs[0,2:4][np.newaxis:,] + \
                global_obs[0,landmarks_idx:landmarks_idx+2*N].reshape(N,2)
            ).reshape(2*N)
        global_obs: ObsType = np.concatenate(
            [all_vel, all_pos, all_landmark_pos]
        )
        return global_obs

def make_ep_spread_env(
    num_agents: int,
    num_nearest_agents: int,
    num_nearest_landmarks: int,
    local_ratio: float = 0.5,
    max_cycles: int = 15,
    is_continuous_action: bool = True
) -> None:
    env: ParallelEnv = simple_spread_v3.parallel_env(render_mode="rgb_array",
                                                    N=num_agents,
                                                    local_ratio=local_ratio,
                                                    max_cycles=max_cycles,
                                                    continuous_actions=is_continuous_action)
    env = LimitLocalObsSimpleSpreadEnv(
        env, num_nearest_agents, num_nearest_landmarks, num_agents
    )
    env = ModifyActionEnv(env)
    env = EPSimpleSpreadEnv(env, num_agents)
    return env