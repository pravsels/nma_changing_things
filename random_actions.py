import numpy as np
from dm_control import suite
import dm_control.suite.swimmer as swimmer
from acme import wrappers
from environment import create_swim_environment
from utils import render, write_video

def test_dm_control(env, n_steps=60):
    """Tests a DeepMind control suite environment by executing random actions."""
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    
    spec = env.action_spec()
    timestep = env.reset()
    frames = [render(env)]

    for _ in range(n_steps):
        action = np.random.uniform(low=spec.minimum, high=spec.maximum, size=spec.shape)
        timestep = env.step(action)
        frames.append(render(env))

    write_video('./random_actions.mp4', frames, verbose=True)

if __name__ == "__main__":
    
    # Load and test the environment
    env = suite.load('swimmer', 'create_swim_environment', task_kwargs={'random': 1})
    test_dm_control(env)

