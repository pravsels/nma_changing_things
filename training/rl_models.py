from tonic.torch import models, normalizers
import torch
import torch.nn as nn
from ncap_swimmer.swimmer_actor import SwimmerActor
from ncap_swimmer.swimmer import SwimmerModule


# ludo nn controller class
class TurnControllerCustom(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(
                49, hidden_size
            ),  # Assuming 5 is the observation size, adjust if needed
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # Output: right, left, speed
        )

    def forward(self, observations):
        #print(f"Observations size in controller: {observations.size()}")
        control = self.network(observations)
        right = torch.tanh(control[..., 0])  # Right turn control
        left = torch.tanh(control[..., 1])  # Left turn control
        speed = torch.tanh(control[..., 2])  # Speed control
        return right, left, speed


def ppo_mlp_model(
    actor_sizes=(64, 64),
    actor_activation=torch.nn.Tanh,
    critic_sizes=(64, 64),
    critic_activation=torch.nn.Tanh,
):
    rl_model = models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_sizes, actor_activation),
            head=models.DetachedScaleGaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )

    return rl_model


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ludo controller
mlp_controller = TurnControllerCustom()
# mlp_controller.to(device)


# dumb controller to check the change in the worm movement
def dumb_controller(o):
    return (torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]))


def ppo_ncap_model(
    n_joints=5,
    action_noise=0.1,
    critic_sizes=(64, 64),
    critic_activation=nn.Tanh,
    **swimmer_kwargs,
):
    rl_model = models.ActorCritic(
        actor=SwimmerActor(
            swimmer=SwimmerModule(n_joints=n_joints, **swimmer_kwargs),
            controller=mlp_controller,
            distribution=lambda x: torch.distributions.normal.Normal(x, action_noise),
        ),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )

    return rl_model


def d4pg_ncap_model(
    n_joints=5,
    critic_sizes=(256, 256),
    critic_activation=nn.ReLU,
    **swimmer_kwargs,
):
    rl_model = models.ActorCriticWithTargets(
        actor=SwimmerActor(
            swimmer=SwimmerModule(n_joints=n_joints, **swimmer_kwargs),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            # These values are for the control suite with 0.99 discount.
            head=models.DistributionalValueHead(-150.0, 150.0, 51),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )

    return rl_model
