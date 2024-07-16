import collections
from dm_control.utils import rewards
import dm_control.suite.swimmer as swimmer
from constants import SWIM_SPEED

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
        return collections.OrderedDict({
            'joints': physics.joints(),
            'body_velocities': physics.body_velocities()
        })

    def get_reward(self, physics):
        """Returns a smooth reward based on forward velocity."""
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.,
            sigmoid='linear'
        )

