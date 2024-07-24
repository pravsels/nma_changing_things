from tonic.torch import models, normalizers
import torch
import torch.nn as nn 
from ncap_swimmer.swimmer_actor import SwimmerActor
from ncap_swimmer.swimmer import SwimmerModule
import tonic

def ppo_mlp_model(
    actor_sizes=(64, 64),
    actor_activation=torch.nn.Tanh,
    critic_sizes=(64, 64),
    critic_activation=torch.nn.Tanh,
):
    rl_model =  models.ActorCritic(
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

def ppo_ncap_model(
        n_joints=5,
        action_noise=0.1,
        critic_sizes=(64, 64),
        critic_activation=nn.Tanh,
        load_pretrained=False,
        **swimmer_kwargs,
):

    if not load_pretrained:
        rl_model = models.ActorCritic(
            actor=SwimmerActor(
                swimmer=SwimmerModule(n_joints=n_joints,excluded_proprioceptors=[1,3,5], 
                                    **swimmer_kwargs),
                distribution=lambda x: torch.distributions.normal.Normal(x, action_noise),
            ),
            critic=models.Critic(
                encoder=models.ObservationEncoder(),
                torso=models.MLP(critic_sizes, critic_activation),
                head=models.ValueHead(),
            ),
            observation_normalizer=normalizers.MeanStd(),
        )
        print(rl_model)
        return rl_model
    else:
        # load pretrained SwimmerModule checkpoint
        swimmer = SwimmerModule(n_joints=n_joints, **swimmer_kwargs)

        # Load pretrained weights
        pretrained_path ='data/local/experiments/tonic/swimmer-swim_task/ppo_ncap_model_128/checkpoints/step_50000.pt'
        pretrained_state_dict = torch.load(pretrained_path)
        swimmer.load_state_dict(pretrained_state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
        
        # Create ActorCritic model
        rl_model = models.ActorCritic(
            actor=SwimmerActor(
                swimmer=swimmer,
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
        actor=SwimmerActor(swimmer=SwimmerModule(n_joints=n_joints, **swimmer_kwargs),),
        critic=models.Critic(
        encoder=models.ObservationActionEncoder(),
        torso=models.MLP(critic_sizes, critic_activation),
        # These values are for the control suite with 0.99 discount.
        head=models.DistributionalValueHead(-150., 150., 51),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )

    return rl_model
