import os
import tonic
import tonic.torch
from training.rl_models import *

def train(header, agent, environment, name='test', trainer='tonic.Trainer()',
          before_training=None, after_training=None, parallel=1, sequential=1, seed=0, load_pretrained=False):
    
    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the train and test environments.
    _environment = environment
    environment = tonic.environments.distribute(lambda: eval(_environment), parallel, sequential)
    test_environment = tonic.environments.distribute(lambda: eval(_environment))

    # Build the agent.
    agent = eval(agent)
    agent.initialize(
        observation_space=test_environment.observation_space,
        action_space=test_environment.action_space, seed=seed)
    if load_pretrained:
        print('Loading pretrained model')
        agent.load('data/local/experiments/tonic/swimmer-swim_task/ppo_ncap_model_128/checkpoints/step_50000')

    # Choose a name for the experiment.
    if hasattr(test_environment, 'name'):
        environment_name = test_environment.name
    else:
        environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if parallel != 1 or sequential != 1:
            name += f'-{parallel}x{sequential}'

    # Initialize the logger to save data to the path environment/name/seed.
    path = os.path.join('data', 'local', 'experiments', 'tonic', environment_name, name)
    tonic.logger.initialize(path, script_path=None, config=args)

    # Build the trainer.
    trainer = eval(trainer)
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
    )
    # Run some code before training.
    if before_training:
        exec(before_training)

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(after_training)

