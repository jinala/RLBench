import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from quad_reactive.quad_r_train import QuadReactiveTrainEnv


class QuadReactiveTestEnv(QuadReactiveTrainEnv):
	def __init__(self):
		self.l = 0.2
		self.tunnel_y0_lim = (0.5, 7.5) # min, max
		self.tunnel_y1_lim = (2.0, 10.0) 
		self.tunnel_l_lim = (1.0, 1.0)
		self.num_tunnels_min = 80
		self.num_tunnels_max = 80 

		self.num_tunnels = 80

		self.x_offset = 3.0
		self.x_start = 0.0

		self.x_lookout = 10.0
		self.y_lookout = 0.6
		
		
		self.dt = 0.05
		self.tol = 0.02
		self.t_max = 0.8
		self.t_min = -0.8

		self.low_state = np.array([0.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, 0.0])
		self.high_state = np.array([150.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

		self.viewer = None

		self.low_action = np.array([-5.])
		self.high_action = np.array([5.])
		self.action_space = spaces.Box(low=self.low_action, high=self.high_action,
									   dtype=np.float32)
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
											dtype=np.float32)

		self.seed()
		self.reset()
