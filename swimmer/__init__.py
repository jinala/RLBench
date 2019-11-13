from gym.envs.registration import register 

register(
	id="SwimmerTrain-v0",
	entry_point="swimmer.swimmer_train:Swimmer4TrainEnv",
	max_episode_steps=5000,
)


register(
	id="SwimmerTest-v0",
	entry_point="swimmer.swimmer_test:Swimmer4TestEnv",
	max_episode_steps=5000,
)

