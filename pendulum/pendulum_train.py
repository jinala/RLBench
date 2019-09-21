import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from gym.envs.classic_control import PendulumEnv

class PendulumTrainEnv(PendulumEnv):
	
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

		self.goal_err = 0.0

		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self,u):
		th, thdot, mass = self.state # th := theta

		g = self.g
		m = mass
		l = 1.
		dt = self.dt

		u = np.clip(u, -self.max_torque, self.max_torque)[0]
		self.last_u = u # for rendering
		costs = angle_normalize(th)**2  + .001*(u**2)

		newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
		newth = th + newthdot*dt
		newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

		terminal = self._terminal()
	   
		self.state = np.array([newth, newthdot, mass])
		self.goal_err = self._goal_error()
		return self._get_obs(), -costs, terminal, {}

	def reset(self):
		low = np.array([np.pi - 0.04, -0.04, self.min_mass])
		high = np.array([np.pi + 0.04, 0.04, self.max_mass])
		self.state = self.np_random.uniform(low=low, high=high)
		self.last_u = None
		return self._get_obs()

	def get_safe_error(self):
		return 0.0

	def get_goal_error(self):
		return self.goal_err

	def _goal_error(self):
		theta, thetadot, mass = self.state
		theta = angle_normalize(theta)
		th_err = 0.0
		if theta > 0.05:
			th_err = theta - 0.05
		if theta < -0.05:
			th_err = -0.05 - theta 

		return th_err 

	def get_dt(self):
		return self.dt

	def _get_obs(self):
		theta, thetadot, mass = self.state
		return np.array([np.cos(theta), np.sin(theta), thetadot])

	def _terminal(self):
		theta, thetadot, mass = self.state
		th = angle_normalize(theta)
		return bool(th < 0.05 and th > -0.05)

	

def angle_normalize(x):
	return (((x+np.pi) % (2*np.pi)) - np.pi)