import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import train
from training.rl_models import ppo_mlp_model
from environment.tasks import swim_task

##################  HYPERPARAMETERS ##################
TOTAL_STEPS = 5e5
SAVE_INTERVAL = 1e5
# Model parameters
actor_sizes = (32, 32)
critic_sizes = (32, 32)
# Experiment name
experiment_name = 'mlp_256'
######################################################

if __name__ == "__main__":

    train(
        header='import tonic.torch',
        agent=f'tonic.torch.agents.PPO(model={ppo_mlp_model.__name__}(actor_sizes={actor_sizes}, critic_sizes={critic_sizes}))',
        environment='tonic.environments.ControlSuite("swimmer-swim_task")',
        name=experiment_name,
        trainer=f'tonic.Trainer(steps=int({TOTAL_STEPS}),save_steps=int({SAVE_INTERVAL}))'
    )

