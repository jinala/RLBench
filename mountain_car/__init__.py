from gym.envs.registration import register 

register(
	id="MountainCarTrain-v0",
	entry_point="mountain_car.mountain_car_train:MountainCarTrainEnv",
	max_episode_steps=300,
)


register(
	id="MountainCarTest-v0",
	entry_point="mountain_car.mountain_car_test:MountainCarTestEnv",
	max_episode_steps=10000,
)

