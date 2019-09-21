from gym.envs.registration import register 

register(
	id="CartPoleTrain-v0",
	entry_point="cartpole.cartpole_train:CartPoleTrainEnv",
	max_episode_steps=250,
    reward_threshold=249.5,
)


register(
	id="CartPoleTest-v0",
	entry_point="cartpole.cartpole_train:CartPoleTrainEnv",
	max_episode_steps=15000,
    reward_threshold=15000,
)

