import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Swimmer4TrainEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self):
		mujoco_env.MujocoEnv.__init__(self, '/afs/csail.mit.edu/u/j/jinala/symdiff/state_machines/swimmer/python/swimmer4.xml', 4)
		utils.EzPickle.__init__(self)

		self.goal_err = 0

	def step(self, a):
		#state = np.concatenate((self.sim.data.qpos, self.sim.data.qvel))
		#print("S:%s:%s"%(str(state.tolist()), str(a.tolist())))

		ctrl_cost_coeff = 0.0001
		xposbefore = self.sim.data.qpos[0]
		self.do_simulation(a, self.frame_skip)
		xposafter = self.sim.data.qpos[0]
		reward_fwd = (xposafter - xposbefore) / self.dt
		reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
		reward = reward_fwd + reward_ctrl
		ob = self._get_obs()
		self.goal_err = 0.0 if xposafter > 10.0 else 10.0 - xposafter
		done = False
		if xposafter > 10.0:
			done = True
		return ob, reward, done, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

	def _get_obs(self):
		qpos = self.sim.data.qpos
		qvel = self.sim.data.qvel
		return np.concatenate([qpos.flat[2:], qvel.flat])

	def reset_model(self):
		self.set_state(
			self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
			self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
		)
		return self._get_obs()

	def get_safe_error(self):
		return 0.0

	def get_goal_error(self):
		return self.goal_err  

	def get_dt(self):
		return self.dt

		