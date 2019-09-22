import math

import numpy as np

import gym

from cartpole.cartpole_train import CartPoleTrainEnv
from gym import spaces

class CartPoleTestEnv(CartPoleTrainEnv):

	def __init__(self):
		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.1
		self.total_mass = (self.masspole + self.masscart)
		self.length = 1.0 # actually half the pole's length
		self.polemass_length = (self.masspole * self.length)
		self.min_action = -1.0
		self.max_action = 1.0
		self.force_mag = 10.0
		self.tau = 0.02  # seconds between state updates
		self.kinematics_integrator = 'euler'

		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4

		# Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
		high = np.array([
			self.x_threshold * 2,
			np.finfo(np.float32).max,
			self.theta_threshold_radians * 2,
			np.finfo(np.float32).max])

		self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
									   shape=(1,), dtype=np.float32)
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)

		self.seed()
		self.viewer = None
		self.state = None

		self.steps_beyond_done = None


