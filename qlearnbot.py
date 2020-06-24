import gym
import numpy as np
import time, math, random
from typing import Tuple
from sklearn.preprocessing import KBinsDiscretizer

env = gym.make("CartPole-v0")
env.reset()
done = False

epsilon = 0.1
num_discrete_bins = (6, 12)

obs_shape = env.observation_space.shape
action_shape = env.action_space.n

upper_bounds = [env.observation_space.high[2], math.radians(50)]
lower_bounds = [env.observation_space.low[2], -math.radians(50)]

Q_table = np.zeros(num_discrete_bins +  (action_shape,))
n_episodes = 500

def discretize(_, __, angle, pole_velocity):
    est = KBinsDiscretizer(n_bins=num_discrete_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))


def policy(state: tuple):
    return np.argmax(Q_table[state])

def new_Q_value(reward: float, state_new: tuple, discount_factor=1) -> float:
    future_optimal_value = np.max(Q_table[state_new])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

def learning_rate(n: int, min_rate=0.1) -> float:
    return max(min_rate, 1.0-math.log10((n+1)/25))

def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))


for episode in range(n_episodes):
    totalreward = 0
    current_state, done = discretize(*env.reset()), False
    while not done:
        action = policy(current_state)
        if np.random.random() < exploration_rate(episode):
            action = env.action_space.sample()
        
        obs, reward, done, info = env.step(action)
        totalreward+=reward
        new_state = discretize(*obs)

        lr = learning_rate(episode)
        learnt_value = new_Q_value(reward, new_state)
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1-lr)*(old_value) + lr*learnt_value
        current_state = new_state
    print(f"Episode: {episode} totalreward: {totalreward}")

current_state, done = discretize(*env.reset()), False
while not done:
    action = policy(current_state)
    obs, reward, done, info = env.step(action)
    new_state = discretize(*obs)
    current_state = new_state
    env.render()