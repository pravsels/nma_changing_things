import numpy as np
from dm_control import suite, mujoco
import dm_control.suite.swimmer as swimmer
from acme import wrappers
from environment import swim_task
from utils import render, write_video

# changing the environment
test_MJCF = """
<mujoco>
    <worldbody>
    
"""


def test_dm_control(env, n_steps=600):
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

    write_video("./random_actions.mp4", frames, verbose=True)


if __name__ == "__main__":
    # Load and test the environment
    env = suite.load("swimmer", "swim_task", task_kwargs={"random": 1})
    # import pdb; pdb.set_trace()
    test_dm_control(env)
