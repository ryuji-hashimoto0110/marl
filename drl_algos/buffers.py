from abc import abstractmethod
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from typing import TypeVar
import warnings

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class RolloutBufferForMAPPO:
    """Rollout Buffer for MAPPO class.

    Rollout buffer is usually used to store experiences while training on-policy RL algorithm.
    This RolloutBufferForMAPPO is for multi-agent PPO algorithm,
        so it store observation and actions of all agents.
    An experience consists of sets of 9;
        - obs: observation of an agent.
        - global_obs: global observation of critic.
        - hidden_state: RNN hidden state of an agent.
        - action: action of an agent.
        - reward: reward of an agent.
        - done: whether the environment ended.
        - log_pi: the logarithmic probability density of the action in the current policy.
        - next_obs: next observation of an agent.
        - next_global_obs: next global observation of critic.
    """
    def __init__(
        self,
        buffer_size: int,
        obs_shape: ndarray,
        global_obs_shape: ndarray,
        action_shape: ndarray,
        hidden_size: int,
        agent_num: int,
        device: torch.device
    ) -> None:
        """initialization.

        Args:
            buffer_size (int): _description_
            obs_shape (ndarray): _description_
            global_obs_shape (ndarray): _description_
            action_shape (ndarray): _description_
            hidden_size (int): _description_
            agent_num (int): _description_
        """
        self.obs_shape: ndarray = obs_shape
        self.global_obs_shape: ndarray = global_obs_shape
        self.action_shape: ndarray = action_shape
        self.hidden_size: int = hidden_size
        self.next_idx: int = 0
        self.buffer_size: int = int(buffer_size)
        self.agent_num: int = int(agent_num)
        self.device: torch.device = device
        self.obses: Tensor = torch.empty(
            (self.buffer_size, agent_num, *obs_shape),
            dtype=torch.float, device=device
        )
        self.global_obses: Tensor = torch.empty(
            (self.buffer_size, *global_obs_shape),
            dtype=torch.float, device=device
        )
        self.hidden_states: Tensor = torch.empty(
            (self.buffer_size, agent_num, hidden_size),
            dtype=torch.float, device=device
        )
        self.actions: Tensor = torch.empty(
            (self.buffer_size, agent_num, *action_shape),
            dtype=torch.float, device=device
        )
        self.rewards: Tensor = torch.empty(
            (self.buffer_size, agent_num, 1), dtype=torch.float, device=device
        )
        self.dones: Tensor = torch.empty(
            (self.buffer_size, agent_num, 1), dtype=torch.float, device=device
        )
        self.log_pis: Tensor = torch.empty(
            (self.buffer_size, agent_num, 1), dtype=torch.float, device=device
        )
        self.next_obses: Tensor = torch.empty(
            (self.buffer_size, agent_num, *obs_shape),
            dtype=torch.float, device=device
        )
        self.next_global_obses: Tensor = torch.empty(
            (self.buffer_size, *global_obs_shape),
            dtype=torch.float, device=device
        )

    def append(
        self,
        obs_dic: dict[AgentID, ObsType],
        global_obs: ObsType,
        hidden_state_dic: dict[AgentID, Tensor],
        action_dic: dict[AgentID, ActionType],
        reward_dic: dict[AgentID, float],
        done_dic: dict[AgentID, bool],
        log_pi_dic: dict[AgentID, float],
        next_obs_dic: dict[AgentID, ObsType],
        next_global_obs: ObsType
    ) -> None:
        """add one experience to the buffer.

        Args:
            obs_dic (dict[AgentID, ObsType]): _description_
            hidden_state_dic (dict[AgentID, Tensor]): _description_
            action_dic (dict[AgentID, ActionType]): _description_
            reward_dic (dict[AgentID, float]): _description_
            done_dic (dict[AgentID, bool]): _description_
            log_pi_dic (dict[AgentID, float]): _description_
            next_obs_dic (dict[AgentID, ObsType]): _description_
        """
        obs_dic: dict[AgentID, Tensor] = self.convert_obs2tensor(obs_dic)
        global_obs: Tensor = self.convert_obs2tensor(global_obs)
        action_dic: dict[AgentID, Tensor] = self.convert_action2tensor(action_dic)
        next_obs_dic: dict[AgentID, Tensor] = self.convert_obs2tensor(next_obs_dic)
        next_global_obs: Tensor = self.convert_obs2tensor(next_global_obs)
        for i, agent_id in enumerate(obs_dic.keys()):
            self.obses[self.next_idx, i].copy_(
                obs_dic[agent_id].view(self.obs_shape)
            )
            self.hidden_states[self.next_idx, i].copy_(
                hidden_state_dic[agent_id].view(self.hidden_size)
            )
            self.actions[self.next_idx, i].copy_(
                action_dic[agent_id].view(self.action_shape)
            )
            self.rewards[self.next_idx, i] = float(reward_dic[agent_id])
            self.dones[self.next_idx, i] = float(done_dic[agent_id])
            self.log_pis[self.next_idx, i] = float(log_pi_dic[agent_id])
            self.next_obses[self.next_idx, i].copy_(
                next_obs_dic[agent_id].view(self.obs_shape)
            )
        self.global_obses[self.next_idx].copy_(
            global_obs.view(self.global_obs_shape)
        )
        self.next_global_obses[self.next_idx].copy_(
            next_global_obs.view(self.global_obs_shape)
        )
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def convert_obs2tensor(
        self, obs: Tensor | ObsType | dict[AgentID, ObsType] | dict[AgentID, Tensor]
    ) -> dict[AgentID, Tensor]:
        """_summary_

        Args:
            obs (dict[AgentID, ObsType]): _description_

        Returns:
            dict[AgentID, Tensor]: _description_
        """
        if isinstance(obs, Tensor):
            return obs
        elif not isinstance(obs, dict):
            return torch.tensor(obs, dtype=torch.float, device=self.device)
        obs_dic: dict[AgentID, ObsType] | dict[AgentID, Tensor] = obs
        for agent_id, obs_i in obs_dic.items():
            if isinstance(obs_i, Tensor):
                continue
            obs_dic[agent_id]: Tensor = torch.tensor(
                obs_i, dtype=torch.float, device=self.device
            )
        return obs_dic

    def convert_action2tensor(
        self, action_dic: dict[AgentID, ActionType]
    ) -> dict[AgentID, Tensor]:
        """_summary_

        Args:
            action_dic (dict[AgentID, ActionType]): _description_

        Returns:
            dict[AgentID, Tensor]: _description_
        """
        for agent_id, action_i in action_dic.items():
            action_dic[agent_id]: Tensor = torch.tensor(
                action_i, dtype=torch.float, device=self.device
            )
        return action_dic

    def is_filled(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """
        return self.next_idx == 0

    def get(self) -> tuple[Tensor]:
        """_summary_

        Returns:
            tuple[Tensor]: _description_
        """
        return (
            self.obses,
            self.global_obses,
            self.hidden_states,
            self.actions,
            self.rewards,
            self.dones,
            self.log_pis,
            self.next_obses,
            self.next_global_obses
        )