import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep

def cartpole():
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    env = gym.make('CartPole-v1')

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    q_table = np.zeros([observation_space, action_space])

    state = env.reset()
    # print(env.observation_space.high)

    for i in range(1, 100001):
        state = env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else: 
                action = np.argmax(q_table[state, :])
            
            next_state, reward, done, info = env.step(action)
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1-alpha)*old_value + alpha*(reward + gamma * next_max)
            q_table[state, action] = new_value

            env.render()
        
        if i%100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")
    print("Training Finished\n")

    # state = env.reset()
    # done = False
    # while not done:
    #     env.render()
    #     action = np.argmax(q_table[state])
    #     next_state, reward, done, info = env.step(action)

 
if __name__ == "__main__":
    cartpole()
