import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control.acrobot import *


class AcrobotTrainEnv(AcrobotEnv):
	def __init__(self):
		self.viewer = None

		self.dt = 0.05
		self.min_mass = 0.2
		self.max_mass = 0.5

		self.min_action = -1.0
		self.max_action = 1.0

		high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
		low = np.array([-1.0, -1.0, -1.0, -1.0, -self.MAX_VEL_1, -self.MAX_VEL_2])

		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
		self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
									   shape=(1,), dtype=np.float32)
		self.state = None
		self.goal_err = 0
		self.seed()

	
	def reset(self):
		low = np.array([-0.05, -0.05, -0.05, -0.05, self.min_mass, self.min_mass])
		high =  np.array([0.05, 0.05, 0.05, 0.05, self.max_mass, self.max_mass])
		self.state = self.np_random.uniform(low=low, high=high)
		return self._get_ob()


	def step(self, a):
		s = self.state
		torque = np.clip(a, self.min_action, self.max_action)[0]

		# Now, augment the state with our force action so it can be passed to
		# _dsdt
		s_augmented = np.append(s, torque)

		ns = rk4(self._dsdt, s_augmented, [0, self.dt])
		# only care about final timestep of integration returned by integrator
		ns = ns[-1]
		ns = ns[:-1]  # omit action
		# ODEINT IS TOO SLOW!
		# ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
		# self.s_continuous = ns_continuous[-1] # We only care about the state
		# at the ''final timestep'', self.dt

		ns[0] = wrap(ns[0], -pi, pi)
		ns[1] = wrap(ns[1], -pi, pi)
		ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
		ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
		self.state = ns
		terminal = self._terminal()
		reward = -1. if not terminal else 0.
		self.goal_err = self._goal_error()
		return (self._get_ob(), reward, terminal, {})

	def _get_ob(self):
		s = self.state
		return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

	def _terminal(self):
		s = self.state
		return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

	def get_safe_error(self):
		return 0.0

	def get_goal_error(self):
		return self.goal_err 
		
	def _goal_error(self):
		s = self.state
		h = -np.cos(s[0]) - np.cos(s[1] + s[0])
		if h < 1.0:
			return 1.0 - h
		return 0.0

	def get_dt(self):
		return self.dt 

	def _dsdt(self, s_augmented, t):
		
		g = 9.8
		a = s_augmented[-1]
		s = s_augmented[:-1]
		m1 = s[-2]
		m2 = s[-1]
		l1 = self.LINK_LENGTH_1
		lc1 = self.LINK_COM_POS_1
		lc2 = self.LINK_COM_POS_2
		I1 = self.LINK_MOI * m1 
		I2 = self.LINK_MOI * m2
		theta1 = s[0]
		theta2 = s[1]
		dtheta1 = s[2]
		dtheta2 = s[3]
		d1 = m1 * lc1 ** 2 + m2 * \
			(l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
		d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
		phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
		phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
			   - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
			+ (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
		if self.book_or_nips == "nips":
			# the following line is consistent with the description in the
			# paper
			ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
				(m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
		else:
			# the following line is consistent with the java implementation and the
			# book
			ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
				/ (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
		ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

		return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0., 0., 0.)

   
