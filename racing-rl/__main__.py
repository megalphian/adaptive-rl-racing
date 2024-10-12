from itertools import count
import gymnasium as gym

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import torch

from dqn.main import DQNManager

from enum import Enum

class MODE(Enum): 
    TRAIN = 0
    TEST = 1

current_mode = MODE.TRAIN

# Initialise the environment
if current_mode == MODE.TRAIN:
    # env = gym.make("CartPole-v1")
    env = gym.make("CarRacing-v3", continuous=False)
elif current_mode == MODE.TEST:
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CarRacing-v3", render_mode="human", continuous=False)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

obs, _ = env.reset()

if current_mode == MODE.TRAIN:

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    manager = DQNManager(env)

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        state = state.permute(2, 0, 1).unsqueeze(0)
        cumulated_reward = 0
        for t in count():
            action = manager.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(np.int64(action.item()))
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device)
                next_state = next_state.permute(2, 0, 1).unsqueeze(0)

            cumulated_reward += reward
            # Store the transition in memory
            action = torch.tensor([[action]], device=device, dtype=torch.long)
            reward = torch.tensor([reward], device=device)
            manager.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            manager.optimize_model()
            manager.soft_update()
            
            if done:
                manager.episode_durations.append(t + 1)
                manager.rewards.append(cumulated_reward)
                if(i_episode % 10 == 0):
                    print(f"Episode {i_episode} lasted {t + 1} steps")
                    stats = manager.get_stats()
                    print(f"Mean duration: {stats[0]}, Mean reward: {stats[1]}")
                break

    print('Complete')

    # save the model
    torch.save(manager.policy_net.state_dict(), "model.pth")

elif current_mode == MODE.TEST:

    manager = DQNManager(env)
    manager.policy_net.load_state_dict(torch.load("Models/Cartpole/model.pth", weights_only=True))
    manager.policy_net.eval()

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        env.render()
        action = manager.policy_net(state).max(1).indices.view(1, 1)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Move to the next state
        state = next_state

        if done:
            break

    print('Complete')
    print(f"Episode lasted {t + 1} steps")

    env.close()