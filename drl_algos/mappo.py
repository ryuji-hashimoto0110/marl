# actorに人数分self.rnn_stateを保持して，
# forwardの引数のrnn_states==Noneの場合はrnn_states=self.rnn_stateにして，出力されたrnn_stateを保持する．
# bufferから取り出された場合はそれを使い，出力されたrnn_stateは保持しない．
# 環境reset時は，initialize_rnn_stateするとか？

# BPTTとhidden_stateのinitialization

from abc import ABC, abstractmethod
from drl_utils import initialize_hidden_states_dic
from drl_utils import initialize_module_orthogonal
from gym import Env
from numpy import ndarray
import pathlib
from pathlib import Path
from pettingzoo.utils.env import ParallelEnv
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from typing import Optional, TypeVar

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class MAPPOActor(Module):
    """_summary_
    """
    def __init__(
        self,
        obs_shape: ndarray,
        hidden_size: int,
        action_shape: ndarray,
        agent_ids: list[AgentID],
        device: torch.device
    ) -> None:
        """_summary_

        Args:
            agent_num (int): _description_
            obs_shape (ndarray): _description_
            hidden_size (int): _description_
            action_shape (ndarray): _description_
        """
        self.hidden_size: int = 64
        self.agent_ids: list[AgentID] = agent_ids
        self.device: torch.device = device
        self.rnnlayer: Module = nn.GRU(
            input_size=obs_shape[0],
            hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        initialize_module_orthogonal(self.rnnlayer)
        self.hidden_states_dic: dict[AgentID, Tensor] = self.initialize_h()
        self.actlayer: Module = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0])
        )
        self.log_stds: Tensor = nn.Parameter(torch.zeros(1, action_shape[0]))

    def initialize_h(self) -> None:
        """_summary_
        stepでresetした時呼ぶ．
        """
        self.hidden_states_dic: dict[AgentID, Tensor] = initialize_hidden_states_dic(
            self.agent_ids, self.hidden_size, self.device
        )

    def forward(
        self,
        obs: dict[AgentID, Tensor],
        hidden_states_dic: Optional[dict[AgentID, Tensor]] = None
    ) -> dict[AgentID, Tensor]:
        is_update_self_h: bool = False
        if hidden_states_dic is None:
            hidden_states_dic: dict[AgentID, Tensor] = self.hidden_states_dic
            is_update_self_h = True
        action: dict[AgentID, Tensor] = {}
        for agent_id in obs.keys():
            obs_i: Tensor = obs[agent_id].unsqueeze_(0).unsqueeze_(0)
            hidden_state_i: Tensor = hidden_states_dic[agent_id]
            act_features, hidden_state: tuple[Tensor] = self.rnnlayer(
                obs_i, hidden_state_i
            )
            if is_update_self_h:
                self.hidden_states_dic[agent_id] = hidden_state
            mean_i = self.actlayer(act_features[:,-1,:])
            action[agent_id] = torch.tanh(mean_i).clamp(-0.999,0.999)
        return action
