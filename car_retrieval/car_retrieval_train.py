
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
#from gym.envs.classic_control import rendering

from car_retrieval.collision import * 

class CarRetrievalTrainEnv(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 30
	}

	def __init__(self):
		self.height = 5
		self.width = 1.8
		self.dist_min = 12.0
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

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		tau = self.tau
		old_obs = self._obs()
		action = np.clip(action, self.low_action, self.high_action)
		v,w = action 
		w = w/10.0

		x,y,ang, dist = self.state   
		beta = np.arctan(0.5*np.tan(w))
		dx = v*np.cos(ang + beta)*tau 
		dy = v*np.sin(ang + beta)*tau 
		da = v/(self.height/2.0)*np.sin(beta)*tau 

		x = x + dx 
		y = y + dy 
		ang = ang + da 

		self.state = np.array([x, y, ang, dist])

		done = bool(x <= self.x_lane_2 - self.width and abs(ang- self.goal_ang) <= 0.05)

		obs = self._obs()
		obs = np.concatenate((obs, old_obs), axis = None)

		self.goal_err = self._goal_error()

		return obs, self._reward(), done, {}

	def reset(self):
		x = self.x_lane_2 + self.np_random.uniform(low = -0.04, high = 0.04)
		ang = np.pi/2.0 + self.np_random.uniform(low = -0.04, high = 0.04)
		dist = self.np_random.uniform(self.dist_min, self.dist_max)
		y = self.height + 0.20
		#print(dist)

		self.state = np.array([x, y, ang, dist])
		obs = self._obs()
		old_obs = self._obs()
		obs = np.concatenate((obs, old_obs), axis = None)
		return obs

	def _obs(self):
		x,y,ang,dist = self.state 
		d1 = 1e20 # min dist to front car
		d2 = 1e20 # min dist to back car

		vertices = get_all_vertices(x, y, ang, self.width, self.height)
		for v in vertices:
			d = max(dist - self.height/2.0 - v[1], self.x_lane_2 - self.width/2.0 - v[0])
			if d < d1: 
				d1 = d 

			d = max(v[1] - self.height/2.0, self.x_lane_2 - self.width/2.0 - v[0])
			if d < d2:
				d2 = d 
		return np.array([x, y, ang, d1, d2])


	def _reward(self):
		e1 = self._check_collision()
		e2 = self._check_boundaries()

		e3 = self._check_goal()

		if e1 + e2 > 0.01:
			return -e3 - 10.0
		else:
			return -e3

	def _check_collision(self):
		x,y,ang,d = self.state

		# obstacle 1
		bx = self.x_lane_2
		by = 0.0
		e1 = check_collision_box(x, y, ang, bx, by, 'l', self.width, self.height)

		# obstacle 2
		bx = self.x_lane_2
		by = d
		e2 = check_collision_box(x, y, ang, bx, by, 'u', self.width, self.height)

		return e1 + e2

	def _check_boundaries(self):
		x,y,ang,_ = self.state

		vertices = get_all_vertices(x, y, ang, self.width, self.height)
		d1 = 1e20 
		d2 = 1e20
		for v in vertices:
			d = 2.5 - v[0]
			if d < d1:
				d1 = d 

			d = v[0] - (-5)
			if d < d2:
				d2 = d 

		err = 0.0
		if d1 < 0.0:
			err += -d1 

		if d2 < 0.0:
			err += -d2 

		return err 
		
		

	def _check_goal(self):
		# unpack
		x,y,ang, dist = self.state

		# error for x
		if (x > self.x_lane_2 - self.width):
			return x - self.x_lane_2 + self.width + 0.1;

		
		# error for ang
		return abs(ang - self.goal_ang)*5.0;


	def get_safe_error(self):
		e1 = self._check_collision()
		e2 = self._check_boundaries()

		return e1 + e2 

	def get_goal_error(self):
		return self.goal_err 
		
	def _goal_error(self):
		x,y,ang, dist = self.state
		error = 0
		# error for x
		if (x > self.x_lane_2 - self.width):
			error += x - self.x_lane_2 + self.width ;

		if (abs(ang - self.goal_ang) > 0.05):
			error += (abs(ang - self.goal_ang) - 0.05)*5.0
		return error 

	def get_dt(self):
		return self.tau


	def render(self, mode='human'):
		screen_width = 600
		screen_height = 600

		world_width = 60.0
		scale = screen_width/world_width
		# unpack state
		x,y,ang,dist = self.state

		# Scales objects
		dist = dist * scale
		w_car = self.width * scale
		h_car = self.height * scale
		dt = self.dt
		
		if self.viewer is None:
			
			# Launches the viewer
			self.viewer = rendering.Viewer(screen_width, screen_height)
			
			# Creates the my car shape
			l,r,t,b = -w_car/2, w_car/2, h_car/2, -h_car/2
			car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.cartrans = rendering.Transform()
			car.add_attr(self.cartrans)
			self.viewer.add_geom(car)
			
			# Creates the stationary car 1
			l,r,t,b = -w_car/2, w_car/2, h_car/2, -h_car/2
			car1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.cartrans1 = rendering.Transform()
			car1.add_attr(self.cartrans1)
			self.cartrans1.set_translation(self.x_lane_2*scale + screen_width/2.0 , screen_height/2.0)
			self.viewer.add_geom(car1)

			# Creates the stationary car 2
			l,r,t,b = -w_car/2, w_car/2, h_car/2, -h_car/2
			car2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.cartrans2 = rendering.Transform()
			car2.add_attr(self.cartrans2)
			self.cartrans2.set_translation(self.x_lane_2*scale + screen_width/2.0, dist + screen_height/2.0)
			self.viewer.add_geom(car2)
			
		
		# Translate and rotate the car
		x = scale * x + screen_width/2.0
		y = scale * y + screen_height/2.0
		ang = ang
		
		self.cartrans.set_translation(x, y)
		self.cartrans.set_rotation(ang - np.pi / 2.0) 
		
		#time.sleep(dt)
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
