# actorに人数分self.rnn_stateを保持して，
# forwardの引数のrnn_states==Noneの場合はrnn_states=self.rnn_stateにして，出力されたrnn_stateを保持する．
# bufferから取り出された場合はそれを使い，出力されたrnn_stateは保持しない．
# 環境reset時は，initialize_rnn_stateするとか？

# BPTTとhidden_stateのinitialization

from abc import ABC, abstractmethod
from buffers import RolloutBufferForMAPPO
from algorithm import Algorithm
from drl_utils import calc_log_pi
from drl_utils import initialize_hidden_state_dic
from drl_utils import initialize_module_orthogonal
from drl_utils import reparametrize
from gym import Env
import numpy as np
from numpy import ndarray
import pathlib
from pathlib import Path
from pettingzoo.utils.env import ParallelEnv
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import Module
from typing import Any, Optional, TypeVar

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class MAPPOActor(Module):
    """MAPPO Actor class.

    Approximate Multi-Agent Proximal Policy Optimization (MAPPO) policy.
    """
    def __init__(
        self,
        obs_shape: ndarray,
        hidden_size: int,
        action_shape: ndarray,
        agent_ids: list[AgentID],
        device: torch.device
    ) -> None:
        """initialization.

        MAPPOActor adopts parameter sharing. Otherwise, each agent keep indivisual hidden state.
        Orthogonal initialization is applied to all network.

        Args:
            obs_shape (ndarray): observation shape of each agent. assume all agents share same observation space.
            hidden_size (int): dimension of hidden state of each agent.
            action_shape (ndarray): action shape of each agent. assume all agents share same action space.
            agent_ids (list[AgentID]): list of agent IDs.
            device (torch.device)
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.agent_ids: list[AgentID] = agent_ids
        self.device: torch.device = device
        self.rnnlayer: Module = nn.GRU(
            input_size=obs_shape[0],
            hidden_size=hidden_size, num_layers=1, batch_first=True
        )
        initialize_module_orthogonal(self.rnnlayer)
        self.initialize_h()
        self.actlayer: Module = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0])
        )
        initialize_module_orthogonal(self.actlayer)
        self.log_std: Tensor = nn.Parameter(torch.zeros(1, action_shape[0]))

    def initialize_h(self) -> None:
        """initialize hidden states of all agents.

        This method is called by algo.step when environment is reset.
        """
        self.hidden_state_dic: dict[AgentID, Tensor] = initialize_hidden_state_dic(
            self.agent_ids, self.hidden_size, self.device
        )

    def calc_mean(
        self,
        obs_dic: dict[AgentID, Tensor],
        hidden_state_dic: Optional[dict[AgentID, Tensor]] = None
    ) -> dict[AgentID, Tensor]:
        """calculate the mean of action distribution.

        Args:
            obs_dic (dict[AgentID, Tensor]): observation dictionary whose key is agent ID and value is observation tensor, obs_i.
                obs_i.shape = (batch_size, *obs_shape).
            hidden_state_dic (Optional[dict[AgentID, Tensor]], optional): hidden state dictionary
                whose key is agent ID and value is hidden state tensor, hidden_state_i.
                hidden_state_i.shape = (batch_size, time_length, hidden_size).time_length must be 1. Defaults to None.

        Returns:
            mean_dic (dict[AgentID, Tensor]): mean dictionary whose key is agent ID and value is mean of action distribution
                before applying output activation function, mean_i. mean_i.shape = (batch_size, *action_shape).
        """
        is_update_self_h: bool = False
        if hidden_state_dic is None:
            hidden_state_dic: dict[AgentID, Tensor] = self.hidden_state_dic
            is_update_self_h = True
        mean_dic: dict[AgentID, Tensor] = {}
        for agent_id, obs_i in obs_dic:
            obs_i: Tensor = obs_i.unsqueeze_(1)
            hidden_state_i: Tensor = hidden_state_dic[agent_id]
            act_feature_i, hidden_state_i: tuple[Tensor] = self.rnnlayer(
                obs_i, hidden_state_i
            )
            if is_update_self_h:
                self.hidden_state_dic[agent_id] = hidden_state_i
            mean_i: Tensor = self.actlayer(act_feature_i[:,-1,:])
            mean_dic[agent_id] = mean_i
        return mean_dic

    def forward(
        self,
        obs_dic: dict[AgentID, Tensor],
        hidden_state_dic: Optional[dict[AgentID, Tensor]] = None
    ) -> dict[AgentID, Tensor]:
        """forward method.

        Return the mean with tanh applied.

        Args:
            obs_dic (dict[AgentID, Tensor]): observation dictionary whose key is agent ID and value is observation tensor, obs_i.
                obs_i.shape = (batch_size, *obs_shape).
            hidden_state_dic (Optional[dict[AgentID, Tensor]], optional): hidden state dictionary
                whose key is agent ID and value is hidden state tensor, hidden_state_i.
                hidden_state_i.shape = (batch_size, time_length, hidden_size). Defaults to None.

        Returns:
            action_dic (dict[AgentID, Tensor]): action dictionary whose key is agent ID and value is action tensor, action_i.
                action_i.shape = (batch_size, *action_shape).
        """
        mean_dic: dict[AgentID, Tensor] = self.calc_mean(obs_dic, hidden_state_dic)
        action_dic: dict[AgentID, Tensor] = {}
        for agent_id, mean_i in mean_dic:
            action_i: Tensor = torch.tanh(mean_i).clamp(-0.999,0.999)
            action_dic[agent_id] = action_i
        return action_dic

    def sample(
        self,
        obs_dic: dict[AgentID, Tensor],
        hidden_state_dic: Optional[dict[AgentID, Tensor]] = None
    ) -> tuple[dict[AgentID, Tensor], dict[AgentID, Tensor]]:
        """sample probabilistic actions.

        Sample from the distribution (diagonal gaussian + tanh)
        using reparametrization trick and calculate the result's probability density.

        Args:
            obs_dic (dict[AgentID, Tensor]): observation dictionary
                whose key is agent ID and value is observation tensor, obs_i.
                obs_i.shape = (batch_size, *obs_shape).
            hidden_state_dic (Optional[dict[AgentID, Tensor]], optional):
                hidden state dictionary whose key is agent ID and value is hidden state tensor, hidden_state_i.
                hidden_state_i.shape = (batch_size, time_length, hidden_size). Defaults to None.

        Returns:
            action_dic (dict[AgentID, Tensor]): action dictionary whose key is agent ID and value is action tensor, action_i.
                action_i.shape = (batch_size, *action_shape).
            log_pi_dic (dict[AgentID, Tensor]]):
                log_pi dictionary whose key is agent ID and value is logarithmic density of current policy.
        """
        mean_dic: dict[AgentID, Tensor] = self.calc_mean(obs_dic, hidden_state_dic)
        action_dic: dict[AgentID, Tensor] = {}
        log_pi_dic: dict[AgentID, Tensor] = {}
        for agent_id, mean_i in mean_dic:
            action_i, log_pi: tuple[Tensor] = reparametrize(mean_i, self.log_std)
            action_i = action_i.clamp(-0.999,0.999)
            action_dic[agent_id] = action_i
            log_pi_dic[agent_id] = log_pi
        return action_dic, log_pi_dic

    def calc_log_pi(
        self,
        obses: Tensor,
        hidden_states: Tensor,
        actions: Tensor
    ) -> Tensor:
        """calculate logarithmic probability of current policy.

        Args:
            obses (Tensor): _description_
            hidden_states (Tensor): _description_
            actions (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        act_features, hidden_states: tuple[Tensor] = self.rnnlayer(obses, hidden_states)
        means: Tensor = self.actlayer(act_features[:,-1,:])
        noises: Tensor = (torch.atanh(actions) - means) / (self.log_std.exp() + 1e-08)
        log_pis: Tensor = calc_log_pi(self.log_std, noises, actions)
        return log_pis

class MAPPOCritic(Module):
    """MAPPO Critic class.

    Estimate the value of the global state.
    """
    def __init__(
        self,
        global_obs_shape: ndarray,
        device: torch.device
    ) -> None:
        """initialization.

        Args:
            global_obs_shape (ndarray): global observation shape.
                The ideas of global observation can be as follows
                    - concatenation of all agents' local observation (CL)
                    - environment-provided global state (EP)
                    - agent-specific global state (AS)
                    - feature-pruned agent-specific global state (FP)
                In MAPPO paper, FP or AS seem to be recommended
                    but it depends on settings of the task.
            device (torch.device)
        """
        super().__init__()
        self.device: torch.device = device
        self.valuelayer: Module = nn.Sequential(
            nn.Linear(global_obs_shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        initialize_module_orthogonal(self.valuelayer)

    def forward(
        self,
        global_obs: Tensor,
    ) -> Tensor:
        """forward method.

        Args:
            global_obs (Tensor): global observation.

        Returns:
            value (Tensor) : _description_
        """
        value: Tensor = self.valuelayer(global_obs)
        return value

class MAPPO(Algorithm):
    """MAPPO algorithm class.
    """
    def __init__(
        self,
        obs_shape: ndarray,
        global_obs_shape: ndarray,
        action_shape: ndarray,
        hidden_size: int,
        agent_ids: list[AgentID],
        device: torch.device,
        seed: int = 1111,
        rollout_length: int = 2048,
        num_updates_per_rollout: int = 10,
        batch_size: int = 1024,
        gamma: float = 0.995,
        lr_actor: float = 5e-04,
        lr_critic: float = 1e-03,
        clip_eps: float = 0.2,
        lmd: float = 0.97,
        max_grad_norm: float = 0.5
    ) -> None:
        """_summary_

        Args:
            obs_shape (ndarray): observation shape of each agent.
            global_obs_shape (ndarray): global observation shape.
            action_shape (ndarray): action shape of each agent.
            hidden_size (int): dimension of hidden state of each agent.
            agent_ids (list[AgentID]):  list of agent IDs.
            device (torch.device)
            seed (int, optional): random seed. Defaults to 1111.
            rollout_length (int, optional): length of a rollout (a sequence of experiences)
                that can be stored in the buffer. (=buffer_size). Defaults to 2048.
            num_updates_per_rollout (int, optional): number of times to update the network
                using one rollout. Defaults to 10.
            batch_size (int, optional): batch size. rollout is processed in mini batch. Defaults to 1024
            gamma (float, optional): discount rate. Defaults to 0.995.
            lr_actor (float, optional): learning rate of self.actor. Defaults to 5e-04.
            lr_critic (float, optional): learning rate of self.critic. Defaults to 1e-03.
            clip_eps (float, optional): the value to clip importance_ratio (pi / pi_old)
                used in Clipped Surrogate Objective. clip_eps is also used in Value Clipping.
                Defaults to 0.2.
            lmd (float, optional): the value to determine how much future TD errors are important
                in Generalized Advantage Estimation (GAE) . Defaults to 0.97.
            max_grad_norm (float, optional):
                threshold to clip the norm of the gradient.
                gradient clipping is used to avoid exploding gradients. Defaults to 0.5.
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.buffer = RolloutBufferForMAPPO(
            buffer_size=rollout_length,
            obs_shape=obs_shape,
            global_obs_shape=global_obs_shape,
            action_shape=action_shape,
            hidden_size=hidden_size,
            agent_ids=agent_ids,
            device=device
        )
        self.actor = MAPPOActor(
            obs_shape=obs_shape,
            hidden_size=hidden_size,
            action_shape=action_shape,
            agent_ids=agent_ids,
            device=device
        ).to(device)
        self.critic = MAPPOCritic(
            global_obs_shape=global_obs_shape,
            hidden_size=hidden_size,
            device=device
        )
        self.optim_actor = optim.Adam(self.actor.parameters(),
                                    lr=lr_actor)
        self.scheduler_actor = optim.lr_scheduler.LambdaLR(
            self.optim_actor,
            lr_lambda=lambda epoch: max(1e-05 / lr_actor, 0.995**(epoch // 50))
        )
        self.optim_critic = optim.Adam(self.critic.parameters(),
                                    lr=lr_critic)
        self.scheduler_critic = optim.lr_scheduler.LambdaLR(
            self.optim_critic,
            lr_lambda=lambda epoch: max(2e-05 / lr_critic, 0.995**(epoch // 50))
        )
        self.batch_size: int = batch_size
        self.device: torch.device = device
        self.gamma: float = gamma
        self.rollout_length: int = num_updates_per_rollout
        self.num_updates_per_rollout: int = num_updates_per_rollout
        self.clip_eps: float = clip_eps
        self.lmd: float = lmd
        self.max_grad_norm: float = max_grad_norm

    def is_ready_to_update(self, current_total_steps: int) -> bool:
        return current_total_steps % self.rollout_length == 0

    def step(
        self,
        env: ParallelEnv,
        obs_dic: dict[AgentID, ObsType],
        current_episode_steps: int,
        current_total_steps: int
    ) -> tuple[dict[AgentID, ObsType], int]:
        current_episode_steps += 1
        hidden_state_dic: dict[AgentID, Tensor] = self.actor.hidden_state_dic
        action_dic, log_pi_dic: tuple[dict[AgentID, ActionType], dict[AgentID, Tensor]] \
            = self.explore(obs_dic)
        next_obs_dic, reward_dic, done_dic, _, _: \
            tuple[dict[AgentID, ObsType], dict[AgentID, float],
                dict[AgentID, bool], dict[AgentID, bool],
                dict[AgentID, Any]] = env.step(action_dic)
        self.buffer.append(
            obs_dic,
            hidden_state_dic,
            action_dic,
            reward_dic,
            done_dic,
            log_pi_dic,
            next_obs_dic
        )
        if sum(done_dic.values()) > 0:
            current_episode_steps = 0
            self.actor.initialize_h()
            next_obs_dic: dict[AgentID, ObsType] = env.reset()
        return next_obs_dic, current_episode_steps

    def convert_obs2tensor(
        self, obs_dic: dict[AgentID, ObsType]
    ) -> dict[AgentID, Tensor]:
        """_summary_

        Args:
            obs (dict[AgentID, ObsType]): _description_

        Returns:
            dict[AgentID, Tensor]: _description_
        """
        for agent_id, obs_i in obs_dic:
            obs_dic[agent_id]: Tensor = torch.tensor(
                obs_i, dtype=torch.float, device=self.device
            ).unsqueeze_(0)
        return obs_dic

    def convert_tensor2action(
        self, action_dic: dict[AgentID, Tensor]
    ) -> dict[AgentID, ActionType]:
        """_summary_

        Args:
            action_dic (dict[AgentID, Tensor]): _description_

        Returns:
            dict[AgentID, ActionType]: _description_
        """
        for agent_id, action_i in action_dic:
            action_dic[agent_id]: ActionType = action_i.detach().cpu().numpy()[0]
        return action_dic

    def update(self) -> None:
        """_summary_
        """
        obses, global_obses, hidden_states, actions, rewards, dones, \
            log_pis_old, next_obses, next_global_obses: \
            tuple[Tensor] = self.buffer_get()
        with torch.no_grad():
            values: Tensor = self.critic(global_obses)
            next_values: Tensor = self.critic(next_global_obses)
        targets, advantages: tuple[Tensor] = self.calc_gae(
            values, rewards, dones, next_values
        )
        for _ in range(self.num_updates_per_rollout):
            indices: ndarray = np.arange(self.rollout_length)
            np.random.shuffle(indices)
            for start in range(0, self.rollout_length, self.batch_size):
                sub_indices: ndarray = indices[start:start+self.batch_size]
                self.update_critic(
                    global_obses[sub_indices], targets[sub_indices]
                )
                self.update_actor(
                    obses[sub_indices],
                    hidden_states[sub_indices],
                    actions[sub_indices],
                    log_pis_old[sub_indices],
                    advantages[sub_indices]
                )
                self.scheduler_actor.step()
                self.scheduler_critic.step()
