import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control import Continuous_MountainCarEnv


class MountainCarTrainEnv(Continuous_MountainCarEnv):
	
	def __init__(self, goal_velocity = 0):
		self.min_action = -1.0
		self.max_action = 1.0
		self.min_position = -1.2
		self.max_position = 0.6
		self.max_speed = 0.07
		self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
		self.goal_velocity = goal_velocity
		self.max_power = 0.0015
		self.min_power = 0.0005

		self.low_state = np.array([self.min_position, -self.max_speed])
		self.high_state = np.array([self.max_position, self.max_speed])

		self.viewer = None

		self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
									   shape=(1,), dtype=np.float32)
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
											dtype=np.float32)
		self.goal_err = 0.0

		self.seed()
		self.reset()


	def step(self, action):

		position = self.state[0]
		velocity = self.state[1]
		power = self.state[2]
		force = min(max(action[0], -1.0), 1.0)

		velocity += force*power -0.0025 * math.cos(3*position)
		if (velocity > self.max_speed): velocity = self.max_speed
		if (velocity < -self.max_speed): velocity = -self.max_speed
		position += velocity
		if (position > self.max_position): position = self.max_position
		if (position < self.min_position): position = self.min_position
		if (position==self.min_position and velocity<0): velocity = 0

		done = bool(position >= self.goal_position)

		reward = 0
		if position < self.goal_position:
			reward += -(self.goal_position - position)
		if done:
			reward += 100.0
		#reward-= math.pow(action[0],2)*0.1

		self.state = np.array([position, velocity, power])
		self.goal_err = self._goal_error()
		return self.state[0:2], reward, done, {}

	def get_safe_error(self):
		return 0.0

	def get_goal_error(self):
		return self.goal_err

	def _goal_error(self):
		position = self.state[0]
		if position < self.goal_position:
			return self.goal_position - position 
		return 0.0

	def get_dt(self):
		return 1.0

	def reset(self):
		power = self.np_random.uniform(low = self.min_power, high = self.max_power)
		pos = self.np_random.uniform(low=-0.6-0.04, high=-0.6+0.04)
		vel = self.np_random.uniform(low=-0.04, high=0.04)
		self.state = np.array([pos, vel, power])
		return np.array(self.state)[0:2]

   