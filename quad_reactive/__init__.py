from gym.envs.registration import register 

register(
	id="QaudReactiveTrain-v0",
	entry_point="quad_reactive.quad_reactive_train:QuadReactiveTrainEnv",
	max_episode_steps=500,
)


register(
	id="QuadReactiveTest-v0",
	entry_point="quad_reactive.quad_reactive_test:QuadReactiveTestEnv",
	max_episode_steps=2000,
)

