import collections
from dm_control.utils import rewards
import dm_control.suite.swimmer as swimmer
from constants import SWIM_SPEED
import numpy as np

class Swim(swimmer.Swimmer):
    """Task to swim forwards at the desired speed."""
    def __init__(self, desired_speed=SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target by setting alpha to 0.
        for target in ['target', 'target_default', 'target_highlight']:
            physics.named.model.mat_rgba[target, 'a'] = 0

    def get_observation(self, physics):
        """Returns an observation of joint angles and body velocities."""
        swimmer_pos = physics.named.data.geom_xpos['head'][:2]
        # Get the sphere's position
        sphere_pos = [0.5, 0.5]  # Assuming the sphere's position is fixed at [0.5, 0.5]
        
        # Calculate the distance between the swimmer and the sphere
        distance = np.linalg.norm(swimmer_pos - sphere_pos)
        print("Distance: ", distance)
        return collections.OrderedDict({
            'joints': physics.joints(),
            'body_velocities': physics.body_velocities(),
            'distance': distance
        })

    def get_reward(self, physics):
        """Returns a smooth reward that is 0 when stopped or moving backwards, 
           and rises linearly to 1 when moving forwards at the desired speed."""
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear'
        )
    
