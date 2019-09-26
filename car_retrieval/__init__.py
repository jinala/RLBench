from gym.envs.registration import register 

register(
	id="CarRetrievalTrain-v0",
	entry_point="car_retrieval.car_retrieval_train:CarRetrievalTrainEnv",
	max_episode_steps=100,
)

register(
	id="CarRetrievalTrain-v1",
	entry_point="car_retrieval.car_retrieval_train1:CarRetrievalTrainEnv1",
	max_episode_steps=100,
)

register(
	id="CarRetrievalTrain-v2",
	entry_point="car_retrieval.car_retrieval_train2:CarRetrievalTrainEnv2",
	max_episode_steps=100,
)

register(
	id="CarRetrievalTrain-v3",
	entry_point="car_retrieval.car_retrieval_train3:CarRetrievalTrainEnv3",
	max_episode_steps=100,
)

register(
	id="CarRetrievalTrain-v4",
	entry_point="car_retrieval.car_retrieval_train4:CarRetrievalTrainEnv4",
	max_episode_steps=100,
)

register(
	id="CarRetrievalTrain-v5",
	entry_point="car_retrieval.car_retrieval_train5:CarRetrievalTrainEnv5",
	max_episode_steps=100,
)

register(
	id="CarRetrievalTest-v0",
	entry_point="car_retrieval.car_retrieval_test:CarRetrievalTestEnv",
	max_episode_steps=1000,
)

