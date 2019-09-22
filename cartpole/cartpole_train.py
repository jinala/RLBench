import math

import numpy as np

import gym

from gym.envs.classic_control import CartPoleEnv
from gym import spaces

class CartPoleTrainEnv(CartPoleEnv):

	def __init__(self):
		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.1
		self.total_mass = (self.masspole + self.masscart)
		self.length = 0.5 # actually half the pole's length
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

		self.safe_error = 0.0

		self.steps_beyond_done = None


	def step(self, action):
		a = min(max(action[0], -1.0), 1.0)
		force = a*self.force_mag

		state = self.state
		x, x_dot, theta, theta_dot = state
		costheta = math.cos(theta)
		sintheta = math.sin(theta)
		temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
		xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
		if self.kinematics_integrator == 'euler':
			x  = x + self.tau * x_dot
			x_dot = x_dot + self.tau * xacc
			theta = theta + self.tau * theta_dot
			theta_dot = theta_dot + self.tau * thetaacc
		else: # semi-implicit euler
			x_dot = x_dot + self.tau * xacc
			x  = x + self.tau * x_dot
			theta_dot = theta_dot + self.tau * thetaacc
			theta = theta + self.tau * theta_dot
		self.state = (x,x_dot,theta,theta_dot)
		done =  theta < -self.theta_threshold_radians \
				or theta > self.theta_threshold_radians
		done = bool(done)
		if done:
			self.safe_error = 1.0
		else:
			self.safe_error = 0.0

		if not done:
			reward = 1.0
		elif self.steps_beyond_done is None:
			# Pole just fell!
			self.steps_beyond_done = 0
			reward = 1.0
		else:
			if self.steps_beyond_done == 0:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0

		return np.array(self.state), reward, done, {}

	def get_safe_error(self):
		return self.safe_error

	def get_goal_error(self):
		return 0.0

	def get_dt(self):
		return self.tau


