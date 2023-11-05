import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from typing import Optional, TypeVar

AgentID = TypeVar("AgentID")

def initialize_module_orthogonal(module: Module) -> None:
    """_summary_

    Args:
        module (Module): _description_
    """
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param)

def initialize_hidden_state_dic(
    agent_ids: list[AgentID],
    hidden_size: int,
    device: torch.device,
    hidden_state_dic: Optional[dict[AgentID, Tensor]]=None
) -> dict[AgentID: Tensor]:
    """_summary_

    Args:
        agent_ids (list[AgentID]): _description_
        hidden_states_dic (dict[AgentID: Tensor]): _description_
        hidden_size (int): _description_
    """
    if hidden_state_dic is not None:
        if sorted(list(hidden_state_dic.keys())) != sorted(agent_ids):
            raise ValueError(
                f"Agent IDs in hidden_states_dic do not consistent with agent_ids"
            )
    else:
        hidden_state_dic: dict[AgentID, Tensor] = {}
    for agent_id in agent_ids:
        hidden_state_dic[agent_id]: Tensor = torch.zeros(1, 1, hidden_size).to(device)
    return hidden_state_dic

def calc_log_pi(
    log_stds: Tensor,
    noises: Tensor,
    actions: Tensor
) -> Tensor:
    gaussian_log_probs: Tensor = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) \
        - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    log_pis: Tensor = \
        gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-10).sum(dim=-1, keepdim=True)
    return log_pis

def reparametrize(
    means: Tensor,
    log_stds: Tensor
) -> tuple[Tensor]:
    stds: Tensor = log_stds.exp()
    noises: Tensor = torch.randn_like(means)
    us: Tensor = means + noises * stds
    actions: Tensor = torch.tanh(us)
    log_pis: Tensor = calc_log_pi(log_stds, noises, actions)
    return actions, log_pis