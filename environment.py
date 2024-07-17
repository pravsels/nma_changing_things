import dm_control.suite.swimmer as swimmer
from dm_control import mjcf
from constants import SWIM_SPEED, DEFAULT_TIME_LIMIT, CONTROL_TIMESTEP, N_LINKS
from swim import Swim
from dm_control.rl import control
from environment_mods import add_changes

@swimmer.SUITE.add()
def swim_task(
    n_links=N_LINKS,
    desired_speed=SWIM_SPEED,
    time_limit=DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={}):
    """Returns the swim task for a n-link swimmer."""

    model_string, assets = swimmer.get_model_and_assets(n_links)

    model = mjcf.from_xml_string(model_string)
    add_changes(model)
    modified_model_string = model.to_xml_string()
    
    physics = swimmer.Physics.from_xml_string(modified_model_string, assets=assets)
    # import pdb; pdb.set_trace()
    task = Swim(desired_speed=desired_speed, random=random)
    
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=CONTROL_TIMESTEP,
        **environment_kwargs
    )

