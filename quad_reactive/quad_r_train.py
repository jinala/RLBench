import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
#from gym.envs.classic_control import rendering

from quad.quad_collision import * 

from quad.quad_train import QuadTrainEnv

class QuadReactiveTrainEnv(QuadTrainEnv):
	def __init__(self):
		self.l = 0.2
		self.tunnel_y0_lim = (0.5, 7.5) # min, max
		self.tunnel_y1_lim = (2.0, 10.0) 
		self.tunnel_l_lim = (1.0, 1.0)
		self.num_tunnels = 40

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

	

	def reset(self):
		x = self.x_start + self.x_offset + self.np_random.uniform(low = -0.04, high = 0.04)
		y = 1.0
		vx = 2.0 + self.np_random.uniform(low = -0.04, high = 0.04)
		vy = 1.0 + self.np_random.uniform(low = -0.04, high = 0.04)
		t = 0.0 + self.np_random.uniform(low = -0.04, high = 0.04)
		w = 0.0 + self.np_random.uniform(low = -0.04, high = 0.04)

		state = [x, y, vx, vy, t, w]

		# sample tunnel
		old_t_y0 = 1.0
		old_t_y1 = 5.0

		y_low = self.np_random.uniform(low = 0.5, high = 2.0)
		y_high = self.np_random.uniform(low = 10.0, high = 12.0)


		delta = 0.8

		increasing = True 
		for i in range(self.num_tunnels):

			t_l = self.tunnel_l_lim[0]
			t_y0 = old_t_y0 + delta if increasing else old_t_y0 - delta
			t_y1 = old_t_y1 + delta if increasing else old_t_y1 - delta

			
			if (not increasing and t_y0 <= y_low)  or (increasing and t_y1 >= y_high) :
				increasing = not increasing
				y_low = self.np_random.uniform(low = 0.5, high = 2.0)
				y_high = self.np_random.uniform(low = 10.0, high = 12.0)

			old_t_y0 = t_y0
			old_t_y1 = t_y1

			state.append(t_y0)
			state.append(t_y1)
			state.append(t_l) 

			if i == 0:
				state[1] = (t_y0 + t_y1)/2.0 + self.np_random.uniform(low = -0.04, high = 0.04)

		self.state = np.array(state)
		return self._obs()

	def _obs(self):
		x,y,vx,vy,t,w = self.state[0:6]
		tunnels = self.state[6:]

		obs = [x, y, vx, vy]

		y_floor = 0 # dist to floor in near neighborhood
		y_roof = 12 # dist to roof in near neighborhood

		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3,3))

		t_start = self.x_start + self.x_offset
		for tunnel in tunnels:
			t_y0, t_y1, t_l = tunnel 
			t_x0 = t_start
			t_x1 = t_start + t_l

			t_start += t_l

			if (x > t_x1) or (x + self.y_lookout < t_x0):
				# out of range
				continue

			if t_y0 > y_floor:
				y_floor = t_y0 

			if t_y1 < y_roof:
				y_roof = t_y1

		obs.append((y - y_floor))
		obs.append((y_roof - y))


		x_floor = 10 # x dist at which floor y > cur y 
		x_roof = 10 # x dist at which roof y < cur y 

		got_x_floor = False 
		got_x_roof = False

		t_start = self.x_start + self.x_offset
		for tunnel in tunnels:
			t_y0, t_y1, t_l = tunnel 
			t_x0 = t_start
			t_x1 = t_start + t_l

			t_start += t_l

			if (x > t_x1) or (x + self.x_lookout < t_x0):
				# out of range
				continue

			if not got_x_floor and t_y0 > y:
				x_floor = max(t_x0 - x, 0)
				got_x_floor = True 

			if not got_x_roof and t_y1 < y:
				x_roof = max(t_x0 - x, 0)
				got_x_roof = True 

		obs.append(x_roof)
		obs.append(x_floor)

		return np.array(obs)

	




