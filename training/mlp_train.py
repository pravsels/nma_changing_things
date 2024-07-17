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
model_size = 128
actor_sizes = (model_size, model_size)
critic_sizes = (model_size, model_size)
# RL model to use 
rl_model_func = ppo_mlp_model
# Experiment name
experiment_name = f'mlp_{model_size}'
######################################################

if __name__ == "__main__":

    train(
        header='import tonic.torch',
        agent=f'tonic.torch.agents.PPO(model={rl_model_func.__name__}(actor_sizes={actor_sizes}, critic_sizes={critic_sizes}))',
        environment='tonic.environments.ControlSuite("swimmer-swim_task")',
        name=experiment_name,
        trainer=f'tonic.Trainer(steps=int({TOTAL_STEPS}),save_steps=int({SAVE_INTERVAL}))'
    )

