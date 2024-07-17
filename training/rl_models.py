from tonic.torch import models, normalizers
import torch

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

