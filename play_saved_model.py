import os
import yaml
import argparse
import numpy as np
import tonic
from moviepy.editor import ImageSequenceClip
from training.rl_models import * 
from environment.tasks import swim_task

def get_checkpoint_path(checkpoint, path):
    if checkpoint == 'none':
        tonic.logger.log('Not loading any weights')
        return None

    cp_path = os.path.join(path, 'checkpoints')
    if not os.path.isdir(cp_path):
        tonic.logger.error(f'{cp_path} is not a directory')
        return None

    checkpoint_ids = [int(file[5:].split('.')[0]) for file in os.listdir(cp_path) if file.startswith('step_')]

    if not checkpoint_ids:
        tonic.logger.error(f'No checkpoint found in {cp_path}')
        return None

    if checkpoint == 'last':
        cp_id = max(checkpoint_ids)
    elif checkpoint == 'first':
        cp_id = min(checkpoint_ids)
    else:
        cp_id = int(checkpoint)
        if cp_id not in checkpoint_ids:
            tonic.logger.error(f'Checkpoint {cp_id} not found in {cp_path}')
            return None

    return os.path.join(cp_path, f'step_{cp_id}')

def play_model(path, checkpoint='last', environment='default', seed=None, header=None):
    """
    Plays a model within an environment and renders the gameplay to a video.

    Parameters:
    - path (str): Path to the directory containing the model and checkpoints.
    - checkpoint (str): Specifies which checkpoint to use ('last', 'first', or a specific ID). 'none' indicates no checkpoint.
    - environment (str): The environment to use. 'default' uses the environment specified in the configuration file.
    - seed (int): Optional seed for reproducibility.
    - header (str): Optional Python code to execute before initializing the model, such as importing libraries.
    """

    # Load the experiment configuration
    with open(os.path.join(path, 'config.yaml'), 'r') as config_file:
        config = argparse.Namespace(**yaml.safe_load(config_file))

    # Run the header first, e.g. to load an ML framework.
    try:
        if config.header:
            exec(config.header)
        if header:
            exec(header)
    except:
        pass

    # Build the agent
    agent = eval(config.agent)

    # Build the environment
    env_func = lambda: eval(config.environment if environment == 'default' else environment)
    environment = tonic.environments.distribute(env_func)
    if seed is not None:
        environment.seed(seed)

    # Initialize the agent
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )

    # Load the weights of the agent from a checkpoint
    checkpoint_path = get_checkpoint_path(checkpoint, path)
    if checkpoint_path:
        agent.load(checkpoint_path)

    # Run the episode
    test_observations = environment.start()
    frames = [environment.render('rgb_array', camera_id=0, width=640, height=480)[0]]
    score, length, steps = 0, 0, 0

    while True:
        # Select an action
        actions = agent.test_step(test_observations, steps)
        assert not np.isnan(actions.sum())

        # Take a step in the environment
        test_observations, infos = environment.step(actions)
        frames.append(environment.render('rgb_array', camera_id=0, width=640, height=480)[0])
        agent.test_update(**infos, steps=steps)

        score += infos['rewards'][0]
        length += 1
        steps += 1

        if infos['resets'][0]:
            break

    # Create and save the video
    video_path = os.path.join(path, 'video.mp4')
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(video_path)

    print(f'Reward for the run: {score}')

    return video_path


if __name__ == "__main__":

    model_folder = 'ppo_ncap_model_32'

    task_name = 'swimmer-swim_task'

    video_path = play_model(f'./data/local/experiments/tonic/{task_name}/{model_folder}', 
                            checkpoint="last", 
                            environment="default")

    print(f"Video saved to: {video_path}")

