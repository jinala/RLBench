import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from pendulum.pendulum_train import PendulumTrainEnv

class PendulumTestEnv1(PendulumTrainEnv):
	
	def __init__(self, g=10.0):
		self.max_speed=8
		self.max_torque=2.
		self.dt=.05
		self.g = g
		self.viewer = None

		self.min_mass = 1.0
		self.max_mass = 1.5

		high = np.array([1., 1., self.max_speed])
		self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

		self.seed()

	def reset(self):
		low = np.array([np.pi - 0.04, -0.04, self.min_mass])
		high = np.array([np.pi + 0.04, 0.04, self.max_mass])
		self.state = self.np_random.uniform(low=low, high=high)
		self.last_u = None
		return self._get_obs()

