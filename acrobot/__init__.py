from gym.envs.registration import register 

register(
	id="AcrobotTrain-v0",
	entry_point="acrobot.acrobot_train:AcrobotTrainEnv",
	max_episode_steps=200,
)


register(
	id="AcrobotTest-v0",
	entry_point="acrobot.acrobot_test:AcrobotTestEnv",
	max_episode_steps=10000,
)

