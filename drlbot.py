import gym
import numpy as np
import torch
import random

class Deep_q_learner(object):
    def __init__(self, state_shape, action_shape, params):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['lr']
        self.best_mean_reward = -float('inf')
        self.best_reward = -float('inf')
        self.training_steps_completed = 0

        if len(self.state_shape) == 1:
            self.DQN = SLP
        elif len(self.state) == 3:
            self.DQN = CNN