import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

from quad.quad_collision import * 

class QuadTrainEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 30
	}

	def __init__(self):
		self.l = 0.2
		self.tunnel_y0_lim = (0.5, 7.5) # min, max
		self.tunnel_y1_lim = (2.0, 10.0) 
		self.tunnel_l_lim = (1.0, 1.0)
		self.num_tunnels = 60

		self.x_offset = 3.0
		self.x_start = 0.0

		self.x_lookout = 10.0
		self.y_lookout = 2.0
		
		
		self.dt = 0.05
		self.tol = 0.02
		self.t_max = 0.8
		self.t_min = -0.8

		self.low_state = np.array([0.0, -10.0, -10.0, -10.0])
		self.high_state = np.array([150.0, 10.0, 10.0, 10.0])

		self.viewer = None

		self.low_action = np.array([-5.])
		self.high_action = np.array([5.])
		self.action_space = spaces.Box(low=self.low_action, high=self.high_action,
									   dtype=np.float32)
		self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
											dtype=np.float32)

		self.goal_err = 0

		self.seed()
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		dt = self.dt
		
		action = np.clip(action, self.low_action, self.high_action)
		ay = action[0]
		ax = 0.0
		alpha = 0.0 

		x,y,vx,vy,t,w = self.state[0:6]
		tunnels = self.state[6: ]

		x = x + vx*dt 
		y = y + vy*dt 
		vx = vx + ax*dt 
		vy = vy + ay*dt 
		t = t + w*dt 
		w = w + alpha*dt  
		
		self.state = np.array([x, y, vx, vy, t, w])
		self.state = np.concatenate((self.state, tunnels), axis = None)

		done = self._done()
		obs = self._obs()
		self.goal_err = self._goal_error()
		return obs, self._reward(), done, {}

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

		delta = 0.8

		increasing = True 
		for i in range(self.num_tunnels):

			t_l = self.tunnel_l_lim[0]
			t_y0 = old_t_y0 + delta if increasing else old_t_y0 - delta
			t_y1 = old_t_y1 + delta if increasing else old_t_y1 - delta

			
			if  t_y0 <= 0.5  or t_y1 >= 10.0 :
				increasing = not increasing


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
		return np.array([x, y, vx, vy])

	def _done(self):
		e = self._check_goal()
		return e < 0.01
		
	def _reward(self):
		e1 = self._check_collision()
		e2 = self._check_copter()

		e3 = self._check_goal()

		e4 = self._get_obj()

		if e1 + e2 > 0.01:
			return -e3 - e4 - 10.0
		else:
			return -e3 - e4

	def _check_collision(self):
		x,y,_,_,t,_ = self.state[0:6]
		tunnels = self.state[6:]

		error = 0.0
		# ground
		e1 = check_collision_with_ground(x, y, t, self.l)
		error += e1
		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3,3))

		tunnel_idx = -1
		start = self.x_start + self.x_offset
		for i in range(len(tunnels)):
			t_yl, t_yu, t_l = tunnels[i]

			t_x0 = start 
			t_x1 = start + t_l 
			if x >= t_x0 and x <= t_x1:
				e2 = check_collision_with_lower_obj(x, y, t, self.l, start, start + t_l, t_yl)
				e3 = check_collision_with_upper_obj(x, y, t, self.l, start, start + t_l, t_yu)
			
				error += e2 
				error += e3

			start += t_l

		return error

	def _check_copter(self):
		# unpack
		x,y,_,vy,t,_ = self.state[0:6]
		tunnels = self.state[6:]

		error = 0
		if (t > self.t_max):
			error += t - self.t_max

		if (t < self.t_min):
			error += self.t_min - t

		return error

	def _check_goal(self):
		# unpack
		x,y,vx,vy,t,w = self.state[0:6]
		tunnels = self.state[6:]

		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3,3))

		goal_x = self.x_start  +self.x_offset 
		for tunnel in tunnels:
			goal_x += tunnel[2]

		error = 0.0
		# error for x
		if (x < goal_x):
			error += goal_x - x

		return error

	def _get_obj(self):
		# try to maximize the distance from obstacles 
		x,y,vx,vy,t,w = self.state[0:6]
		tunnels = self.state[6:]

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

		return abs(y - (y_roof + y_floor)/2.0)


	def get_safe_error(self):
		e1 = self._check_collision()
		e2 = self._check_copter()

		return e1 + e2 

	def get_goal_error(self):
		return self.goal_err
		
	def _goal_error(self):
		return self._check_goal()


	def get_dt(self):
		return self.dt


	def render(self, mode='human'):
		"""
		Renders the state in the viewer using openai gym
		"""
		
		# Gets scaling factors between world and screen
		screen_width = 1200
		screen_height = 250
		
		world_size = 100.0
		
		scale = screen_width / world_size
		
		# unpack state
		ns = np.copy(self.state)
		ns = np.multiply(ns, scale)
		x,y,vx,vy,t,w = ns[0:6]
		tunnels = ns[6:]
		t = t/scale
		t_start= (self.x_start + self.x_offset)*scale 

		# Scales objects
		ql = self.l * 2.0*scale
		qw = 0.1 * scale
		dt = self.dt


		
		if self.viewer is None:
			
			# Launches the viewer
			self.viewer = rendering.Viewer(screen_width, screen_height)
			
			# Creates the my car shape
			l,r,t,b = -ql/2, ql/2, qw/2, -qw/2
			copter = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.coptertrans = rendering.Transform()
			copter.add_attr(self.coptertrans)
			self.viewer.add_geom(copter)

			tunnels = np.array(tunnels)
			tunnels = np.reshape(tunnels, (len(tunnels)//3, 3))

			for tunnel in tunnels:
				t_y0, t_y1, t_l = tunnel
				# Creates the obstacle1
				l,r,t,b = t_start, t_start + t_l, 0.0, t_y0
				block1 = rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], True)
				self.viewer.add_geom(block1)

				# Creates the obstacle1
				l,r,t,b = t_start, t_start + t_l, t_y1, screen_height
				block2 = rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], True)
				self.viewer.add_geom(block2)

				t_start += t_l
			
		
		# Translate and rotate the car	
		self.coptertrans.set_translation(x, y)
		self.coptertrans.set_rotation(t)
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
