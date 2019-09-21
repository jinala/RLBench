from gym.envs.registration import register 

register(
	id="PendulumTrain-v0",
	entry_point="pendulum.pendulum_train:PendulumTrainEnv",
	max_episode_steps=200,
)


register(
	id="PendulumTest-v0",
	entry_point="pendulum.pendulum_test:PendulumTestEnv",
	max_episode_steps=50000,
)


register(
	id="PendulumTest-v1",
	entry_point="pendulum.pendulum_test1:PendulumTestEnv1",
	max_episode_steps=50000,
)
