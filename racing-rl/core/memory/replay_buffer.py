from collections import namedtuple, deque
import random

import matplotlib.pyplot as plt
import numpy as np

import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        transition = Transition(*args)
        self.memory.append(transition)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), np.ones(batch_size), None

    def __len__(self):
        return len(self.memory)