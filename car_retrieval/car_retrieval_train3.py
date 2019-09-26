
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from car_retrieval.car_retrieval_train import * 


class CarRetrievalTrainEnv3(CarRetrievalTrainEnv):

	def __init__(self):
		self.height = 5
		self.width = 1.8
		self.dist_min = 11.5
		self.dist_max = 13.5
		self.x_lane_1 = -1.5
		self.x_lane_2 = 1.0
		self.x_min = -5.0
		self.x_max = 2.5
		self.goal_ang = np.pi/2.0
		self.tau = 0.1

		self.low_state = np.array([self.x_min, 0.0, 0.0, -50.0, -50.0, self.x_min, 0.0, 0.0, -50.0, -50.0])
		self.high_state = np.array([self.x_max, 30.0, np.pi, 50.0, 50.0, self.x_max, 30.0, np.pi, 50.0, 50.0])

		self.viewer = None

		self.low_action = np.array([-5., -5.])
		self.high_action = np.array([5., 5.])
		self.action_space = spaces.Box(low=self.low_action, high=self.high_action,
									   dtype=np.float32)
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
											dtype=np.float32)

		self.goal_err = 0

		self.seed()
		self.reset()

	
