from gym.envs.registration import register 

register(
	id="CarRetrievalTrain-v0",
	entry_point="car_retrieval.car_retrieval_train:CarRetrievalTrainEnv",
	max_episode_steps=100,
)


register(
	id="CarRetrievalTest-v0",
	entry_point="car_retrieval.car_retrieval_test:CarRetrievalTestEnv",
	max_episode_steps=1000,
)

