import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from mountain_car.mountain_car_train import MountainCarTrainEnv


class MountainCarTestEnv(MountainCarTrainEnv):
   
	def __init__(self, goal_velocity = 0):
		self.min_action = -1.0
		self.max_action = 1.0
		self.min_position = -1.2
		self.max_position = 0.6
		self.max_speed = 0.07
		self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
		self.goal_velocity = goal_velocity
		self.max_power = 0.0005
		self.min_power = 0.0003

		self.low_state = np.array([self.min_position, -self.max_speed])
		self.high_state = np.array([self.max_position, self.max_speed])

		self.viewer = None

		self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
									   shape=(1,), dtype=np.float32)
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
											dtype=np.float32)

		self.seed()
		self.reset()


   

   