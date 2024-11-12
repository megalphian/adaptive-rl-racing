from ddpg.ddpg import Actor_CNN, Critic_CNN, Actor, Critic
from core.replay_buffer import ReplayMemory, Transition
from core.noise_generator import NoiseGenerator
from core.envs import EnvMode

from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
import numpy as np

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 1e-3

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
cpu_device = torch.device("cpu")

class DDPGManager:

    def __init__(self, env, env_type):
        self.env = env
        # Get number of actions from gym action space
        self.n_actions = env.action_space.shape[0]
        if env_type == EnvMode.RACING:
            self.n_actions -= 1

        # Get the number of state observations
        state, info = env.reset()
        n_observations = state.shape

        self.memory = ReplayMemory(100000)
        self.env_type = env_type

        if env_type == EnvMode.RACING:
            self.actor_net = Actor_CNN(n_observations, self.n_actions).to(device)
            self.actor_target = Actor_CNN(n_observations, self.n_actions).to(device)
            self.critic_net = Critic_CNN(n_observations, self.n_actions).to(device)
            self.critic_target = Critic_CNN(n_observations, self.n_actions).to(device)
        elif env_type == EnvMode.PENDULUM:
            n_observations = n_observations[0]
            self.actor_net = Actor(n_observations, self.n_actions).to(device)
            self.actor_target = Actor(n_observations, self.n_actions).to(device)
            self.critic_net = Critic(n_observations, self.n_actions).to(device)
            self.critic_target = Critic(n_observations, self.n_actions).to(device)
        else:
            raise ValueError("Environment not supported")
        
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.actor_optimizer = optim.AdamW(self.actor_net.parameters(), lr=LR, amsgrad=True)
        self.actor_loss = deque(maxlen=100)
        
        self.critic_target.load_state_dict(self.critic_net.state_dict())
        self.critic_optimizer = optim.AdamW(self.critic_net.parameters(), lr=LR, amsgrad=True)
        self.critic_loss = deque(maxlen=100)

        self.episode_durations = deque(maxlen=100)
        self.rewards = deque(maxlen=100)

        self.steps_done = 0

        noise_mean = np.full(self.n_actions, 0.0, np.float32)
        noise_std  = np.full(self.n_actions, 0.2, np.float32)
        self.noise_generator = NoiseGenerator(noise_mean, noise_std)

    def decode_model_output(self, model_out):
        return np.array([model_out[0], model_out[1].clip(0, 1), -model_out[1].clip(-1, 0)])
    
    def encode_model_output(self, actions):
        # actions is a tensor of shape (batch_size, n_actions)
        # actions[:, 1] is the throttle value
        # actions[:, 2] is the brake value
        # We need to encode the throttle and brake values into a single action value
        return torch.stack([actions[:, 0], actions[:, 1] - actions[:, 2]], dim=1)

    def select_greedy_action(self, state):
        state = state.to(device)
        with torch.no_grad():
            action = self.actor_net(state).cpu()
        
        if self.env_type == EnvMode.RACING:
            action = self.decode_model_output(action)
        
        return action

    def select_action(self, state):
        self.steps_done += 1
        state = state.to(device)
        with torch.no_grad():
            action = self.actor_net(state).cpu()
        
        action = action.squeeze().numpy()
        
        noise = self.noise_generator.generate()
        action = action + noise

        if self.env_type == EnvMode.RACING:
            action = self.decode_model_output(action)

        return torch.tensor(action, device=cpu_device, dtype=torch.float32)
        
    def get_stats(self):
        # Get the mean duration and rewards of the last 100 episodes
        mean_duration = sum(self.episode_durations)/len(self.episode_durations)
        mean_reward = sum(self.rewards)/len(self.rewards)
        mean_critic_loss = sum(self.critic_loss)/len(self.critic_loss)
        mean_actor_loss = sum(self.actor_loss)/len(self.actor_loss)
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
                                                    if s is not None]).to(device)
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.stack(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        action_batch = self.encode_model_output(action_batch)

        # Compute Q(s_t, a) using critic network
        state_action_values = self.critic_net(state_batch, action_batch)

        # Compute Q'(s_{t+1}, a) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            target_val = self.critic_target(non_final_next_states, self.actor_target(non_final_next_states))
            next_state_values[non_final_mask] = target_val.squeeze()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        state_action_values = state_action_values.squeeze()

        critic_loss = (state_action_values - expected_state_action_values).pow(2).mean()
        self.critic_loss.append(critic_loss)

        # Optimize the model
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_value_(self.critic_net.parameters(), 100)
        self.critic_optimizer.step()

        # Compute actor loss
        # Source: https://github.com/lzhan144/Solving-CarRacing-with-DDPG/blob/master/DDPG.py
        # The actor is trained to maximize the expected return of the critic  
        actor_loss = -(self.critic_net(state_batch, self.actor_net(state_batch)).mean())
        self.actor_loss.append(actor_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_value_(self.actor_net.parameters(), 100)
        self.actor_optimizer.step()

    def soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′

        for target_param, param in zip(self.actor_target.parameters(), self.actor_net.parameters()):
            target_param.data = TAU * param.data + (1 - TAU) * target_param.data

        for target_param, param in zip(self.critic_target.parameters(), self.critic_net.parameters()):
            target_param.data = TAU * param.data + (1 - TAU) * target_param.data

    def save_model(self):
        # save the model
        torch.save(self.actor_net.state_dict(), 'actor.pth')
        torch.save(self.critic_net.state_dict(), 'critic.pth')
