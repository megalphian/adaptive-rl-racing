from ddpg.ddpg import Actor_CNN, Critic_CNN
from core.replay_buffer import ReplayMemory, Transition

import torch
import torch.nn as nn
import torch.optim as optim

import math
import random

import matplotlib.pyplot as plt

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DDPGManager:

    def __init__(self, env):
        self.env = env
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = env.reset()
        n_observations = state.shape

        self.memory = ReplayMemory(10000)

        self.actor_net = Actor_CNN(n_observations, n_actions).to(device)
        self.actor_target = Actor_CNN(n_observations, n_actions).to(device)
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.actor_optimizer = optim.AdamW(self.actor_net.parameters(), lr=LR, amsgrad=True)
        self.actor_loss = []

        self.critic_net = Critic_CNN(n_observations, n_actions).to(device)
        self.critic_target = Critic_CNN(n_observations, n_actions).to(device)
        self.critic_target.load_state_dict(self.critic_net.state_dict())
        self.critic_optimizer = optim.AdamW(self.critic_net.parameters(), lr=LR, amsgrad=True)
        self.critic_loss = []

        self.episode_durations = []
        self.rewards = []

        self.steps_done = 0

    def select_greedy_action(self, state):
        with torch.no_grad():
            return self.actor_net(state).max(1)[1].view(1, 1)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.actor_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def get_stats(self):
        # Get the mean duration and rewards of the last 100 episodes
        mean_duration = sum(self.episode_durations[-100:])/len(self.episode_durations[-100:])
        mean_reward = sum(self.rewards[-100:])/len(self.rewards[-100:])
        mean_critic_loss = sum(self.critic_loss[-100:])/len(self.critic_loss[-100:])
        mean_actor_loss = sum(self.actor_loss[-100:])/len(self.actor_loss[-100:])
        return mean_duration, mean_reward, mean_critic_loss, mean_actor_loss

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) using critic network
        state_action_values = self.critic_net(state_batch, action_batch)

        # Compute Q'(s_{t+1}, a) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.critic_target(non_final_next_states, self.actor_target(non_final_next_states)).squeeze()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        critic_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.critic_loss.append(critic_loss.item())

        # Optimize the model
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), 100)
        self.critic_optimizer.step()

        # Compute actor loss
        # Source: https://github.com/lzhan144/Solving-CarRacing-with-DDPG/blob/master/DDPG.py
        # The actor is trained to maximize the expected return of the critic  
        actor_loss = -self.critic_net(state_batch, self.actor_net(state_batch)).mean()
        self.actor_loss.append(actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.actor_net.parameters(), 100)
        self.actor_optimizer.step()

    def soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def hard_update(self):
        # Hard update of the target network's weights
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        # save the model
        torch.save(self.actor_net.state_dict(), 'actor.pth')
        torch.save(self.critic_net.state_dict(), 'critic.pth')
