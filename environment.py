import dm_control.suite.swimmer as swimmer
from constants import SWIM_SPEED, DEFAULT_TIME_LIMIT, CONTROL_TIMESTEP
from swim_task import Swim
from dm_control.rl import control

@swimmer.SUITE.add()
def create_swim_environment(
    n_links: int = 6,
    desired_speed: float = SWIM_SPEED,
    time_limit: float = DEFAULT_TIME_LIMIT,
    random = None,
    environment_kwargs: dict = {}):
    """Creates and returns the Swim environment."""
    model_string, assets = swimmer.get_model_and_assets(n_links)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    task = Swim(desired_speed=desired_speed, random=random)
    
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=CONTROL_TIMESTEP,
        **environment_kwargs
    )

