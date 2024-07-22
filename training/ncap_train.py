import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import train
from training.rl_models import ppo_ncap_model, d4pg_ncap_model
from environment.tasks import swim_task
from constants import N_LINKS

##################  HYPERPARAMETERS ##################
TOTAL_STEPS = 1e5
SAVE_INTERVAL = 1e4
# Model parameters
model_size = 128
critic_sizes = (model_size, model_size)
# RL model to use
rl_model_func = ppo_ncap_model  # ppo_ncap_model / d4pg_ncap_model
# Experiment name
experiment_name = f"{rl_model_func.__name__}_{model_size}"
######################################################

if __name__ == "__main__":
    train(
        header="import tonic.torch",
        agent=f"tonic.torch.agents.{rl_model_func.__name__.split('_')[0].upper()}(model={rl_model_func.__name__}(n_joints={N_LINKS-1},critic_sizes={critic_sizes}))",
        environment='tonic.environments.ControlSuite("swimmer-swim_task", time_feature=True)',
        name=experiment_name,
        trainer=f"tonic.Trainer(steps=int({TOTAL_STEPS}),save_steps=int({SAVE_INTERVAL}))",
    )
