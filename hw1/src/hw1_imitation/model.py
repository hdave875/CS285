"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from pyexpat import model
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, chunk_size*action_dim))
        self.net = nn.Sequential(*layers)
    def forward (
        self,
        x,
    )->torch.Tensor:
        return self.net(x)
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
            predicted_actions = self.forward(state).view(*action_chunk.shape)
            # p_sample = predicted_actions.detach().cpu().numpy()
            # t_sample = action_chunk.detach().cpu().numpy()
            # print("DEBUG: ACTION CHUNK COMPARISON")
            # print(f"Target (First 2 steps):    {t_sample}")
            # print(f"Predicted (First 2 steps): {p_sample}")
            loss = nn.functional.mse_loss(predicted_actions, action_chunk)
            return loss
       
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10, 
    ) -> torch.Tensor: 
            self.eval()
            with torch.no_grad():
                output = self.forward(state)
                set_of_actions = output.view(-1, self.chunk_size, self.action_dim)
                requested_steps = num_steps
                action_chunk = set_of_actions[:, :requested_steps, :]
                # print(f"State input shape: {state.shape}")
                # print(f"Full output shape: {set_of_actions.shape}")
                # print(f"Returning shape:    {action_chunk.shape}")
                # print(f"Returning type:     {type(action_chunk)}")
                #action_chunk = set_of_actions[0, :num_steps, :]
                #set_of_actions = self.forward(state).flatten()[:num_steps].view(num_steps//2, 2)
            return action_chunk

class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        time_dim: int = 1,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
       
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim + (action_dim * chunk_size) + time_dim, hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[0], hidden_dims[1]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[1], action_dim* chunk_size)
        # )
        self.time_net = nn.Sequential(
            nn.Linear(time_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
        )
        self.state_action_net = nn.Sequential(
            nn.Linear(state_dim + (action_dim * chunk_size), hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.velocity_net = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim * chunk_size)
        )
     
    def forward (
        self,
        state: torch.Tensor,
        action_noisy: torch.Tensor,
        t: torch.Tensor,
    )->torch.Tensor:
        t_emb = self.time_net(t)
        # print(f"Time embedding shape: {t_emb.shape}")
        t_emb = t_emb.squeeze(1)
        sa_input = torch.cat([state, action_noisy.flatten(start_dim=1)], dim=1)
        # print(f"State-Action input shape: {sa_input.shape}")
        sa_emb = self.state_action_net(sa_input)
        # print(f"State-Action embedding shape: {sa_emb.shape}")
        combined = t_emb + sa_emb
        # print(f"Combined embedding shape: {combined.shape}")
        predicted_velocity = self.velocity_net(combined)
        # print(f"Raw predicted velocity shape: {predicted_velocity.shape}")
        predicted_velocity = predicted_velocity.view(action_noisy.shape) 
        #print(f"Forward called with state: {state.shape}, action_noisy: {action_noisy.shape}, t: {t.shape}")
        # action_noisy_flat = action_noisy.flatten(start_dim=1)
        # t_flat = t.flatten(start_dim=1)
        # print(f"Forward called with state: {state.shape}, action_noisy: {action_noisy_flat.shape}, t: {t_flat.shape}")
        # inputs = torch.cat([state, action_noisy_flat, t_flat], dim=1)
        # #print(f"Passing Inputs: {inputs.shape}")
        # predicted_velocity = self.net(inputs)
        # predicted_velocity = predicted_velocity.view(action_noisy.shape)
        return predicted_velocity
    
    def interpolate(
        self,
        t: torch.Tensor,
        action_noisy: torch.Tensor,
        action_expert: torch.Tensor,
    ) -> torch.Tensor:
        #print(f"t: {t.shape}, action_noisy: {action_noisy.shape}, action_expert: {action_expert.shape}")
        action_tau = t*action_expert + (1-t)*action_noisy
        return action_tau
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action_expert: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        tau = torch.rand(B, 1, device=state.device)
        tau = tau.view(B, 1, 1)
        action_noisy = torch.randn_like(action_expert)
        action_tau = self.interpolate(tau, action_noisy, action_expert)
        predicted_velocity = self.forward(state, action_tau, tau)
        actual_velocity = action_expert - action_noisy
        #actual_velocity_flat = actual_velocity.flatten(start_dim=1)
        #print(f"Predicted Velocity: {predicted_velocity.shape}, Actual Velocity: {actual_velocity_flat.shape}")
        loss = nn.functional.mse_loss(predicted_velocity, actual_velocity)
        return loss
    
    def euler_integrate(
        self,
        state: torch.Tensor,
        action_initial: torch.Tensor,
        num_steps: int,
        dt: float = 0.1,
    ) -> torch.Tensor:
        action_tau = action_initial
        for step in range(num_steps):
            t = torch.full((state.shape[0], 1), step * dt, device=state.device)
            velocity = self.forward(state, action_tau, t)
            #print(f"Step {step}: t={t.flatten()[0]:.2f}, velocity shape={velocity.shape}, action_tau shape={action_tau.shape}")
            action_tau = action_tau + velocity * dt
        return action_tau
    
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            action_initial = torch.randn(state.shape[0], self.chunk_size, self.action_dim, device=state.device)
            output = self.euler_integrate(state, action_initial, num_steps, dt=1.0 / num_steps)
            set_of_actions = output.view(-1, self.chunk_size, self.action_dim)
            #requested_steps = num_steps
            #action_chunk = set_of_actions[:, :requested_steps, :]
        return set_of_actions


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
