from gym.envs.registration import register 

register(
	id="CartPoleTrain-v0",
	entry_point="cartpole.cartpole_train:CartPoleTrainEnv",
	max_episode_steps=200,
    reward_threshold=195.0,
)


register(
	id="CartPoleTest-v0",
	entry_point="cartpole.cartpole_train:CartPoleTrainEnv",
	max_episode_steps=10000,
    reward_threshold=9999,
)

