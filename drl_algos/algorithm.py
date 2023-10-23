from abc import ABC, abstractmethod
from gym import Env
from numpy import ndarray
from pettingzoo.utils.env import ParallelEnv
import torch
from torch import Tensor
from typing import TypeVar

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = TypeVar("AgentID")

class Algorithm(ABC):
    """Algorithm class.

    Any RL algorithms are implemented by inheriting from this "Algorithm" class.
    RL algorithm must contains
        self.actor (nn.Module)
            .actor.forward(obs: Tensor | dict[AgentID, Tensor]) -> Tensor | dict[AgentID, Tensor]
            .actor.sample(obs: Tensor | dict[AgentID, Tensor]) -> tuple[Tensor | dict[AgentID, Tensor], float | ndarray]
    and often contains
        self.critic (nn.Module)
            .critic.forward(
                obs: 
                actions: 
            ) -> 
    """
    @torch.no_grad()
    def explore(
        self,
        obs: ObsType | tuple[dict[AgentID, ObsType]]
    ) -> tuple[ActionType | dict[AgentID, ActionType], float | ndarray]:
        """_summary_

        Args:
            obs (ObsType | tuple[dict[AgentID, ObsType]]): _description_

        Returns:
            tuple[ActionType | dict[AgentID, ActionType], float | ndarray]: _description_
        """
        obs: Tensor | dict[AgentID, Tensor] = self.convert_obs2tensor(obs)
        action, log_pi: tuple[Tensor | dict[AgentID, Tensor], float | ndarray] \
            = self.actor.sample(obs)
        action: ActionType | dict[AgentID, ActionType] \
            = self.convert_tensor2action(action)
        return action, log_pi

    @torch.no_grad()
    def exploit(
        self,
        obs: ObsType | tuple[dict[AgentID, ObsType]]
    ) -> ActionType | dict[AgentID, ActionType]:
        obs: Tensor | dict[AgentID, Tensor] = self.convert_obs2tensor(obs)
        action: Tensor | dict[AgentID, Tensor] = self.actor(obs)
        action: ActionType | dict[AgentID, ActionType] \
            = self.convert_tensor2action(action)
        return action

    def convert_obs2tensor(
        self,
        obs: ObsType | tuple[dict[AgentID, ObsType]]
    ) -> Tensor | dict[AgentID, Tensor]:
        """_summary_

        Args:
            obs (ObsType | tuple[dict[AgentID, ObsType]]): _description_

        Returns:
            Tensor | dict[AgentID, Tensor]: _description_
        """
        return torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze_(0)

    def convert_tensor2action(
        self,
        action: Tensor | dict[AgentID, Tensor]
    ) -> ActionType | dict[AgentID, ActionType]:
        """_summary_

        Args:
            action (Tensor | dict[AgentID, Tensor]): _description_

        Returns:
            ActionType | dict[AgentID, ActionType]: _description_
        """
        return action.detach().cpu().numpy()[0]

    @abstractmethod
    def is_ready_to_update(self, current_total_steps: int) -> bool:
        pass

    @abstractmethod
    def step(
        self,
        env: Env | ParallelEnv,
        obs: ObsType | tuple[dict[AgentID, ObsType]],
        current_episode_steps: int,
        current_total_steps: int
    ) -> tuple[ObsType | tuple[dict[AgentID, ObsType]], int]:
        pass

    @abstractmethod
    def update(self):
        pass
