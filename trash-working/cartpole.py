import gym
env = gym.make('CartPole-v1')

env.reset()
for _ in range(1000):
    env.render()
    obs, rew, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
       env.reset()
env.close()
