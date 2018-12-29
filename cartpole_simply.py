import gym
import numpy as np

def basic_policy(obs):
  angle = obs[2]
  return 0 if angle < 0 else 1

env = gym.make("CartPole-v0")
totals = []
for epoch in range(500):
  epoch_rewards = 0
  obs = env.reset()
  for step in range(1000):
    action = basic_policy(obs)
    obs, reward, done, info = env.step(action)
    epoch_rewards += reward
    if done:
      break
  totals.append(epoch_rewards)

print(np.mean(totals), np.min(totals), np.max(totals))
