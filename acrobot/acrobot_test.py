import gym
from gym import spaces
from gym.utils import seeding
import numpy as np 

from acrobot.acrobot_train import AcrobotTrainEnv


class AcrobotTestEnv(AcrobotTrainEnv):
	def __init__(self):
		self.viewer = None

		self.dt = 0.05
		self.min_mass = 0.5
		self.max_mass = 2.0

		self.min_action = -1.0
		self.max_action = 1.0

		high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
		low = np.array([-1.0, -1.0, -1.0, -1.0, -self.MAX_VEL_1, -self.MAX_VEL_2])

		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
		self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
									   shape=(1,), dtype=np.float32)
		self.state = None
		self.goal_err = 0
		self.seed()
