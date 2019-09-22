from gym.envs.registration import register 

register(
	id="QuadReactiveTrain-v0",
	entry_point="quad_reactive.quad_r_train:QuadReactiveTrainEnv",
	max_episode_steps=500,
)


register(
	id="QuadReactiveTest-v0",
	entry_point="quad_reactive.quad_r_test:QuadReactiveTestEnv",
	max_episode_steps=2000,
)

