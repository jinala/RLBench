from gym.envs.registration import register 

register(
	id="QuadTrain-v0",
	entry_point="quad.quad_train:QuadTrainEnv",
	max_episode_steps=500,
)


register(
	id="QuadTest-v0",
	entry_point="quad.quad_test:QuadTestEnv",
	max_episode_steps=2000,
)

