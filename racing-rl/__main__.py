from itertools import count
import gymnasium as gym
import gymnasium.wrappers as gym_wrap

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import torch

from dqn.main import DQNManager
from ddpg.main import DDPGManager

from core.skip_frame import SkipFrame

from enum import Enum

import wandb

class MODE(Enum): 
    TRAIN = 0
    TEST = 1

class Policy(Enum):
    DQN = 0
    DDPG = 1

current_mode = MODE.TEST
current_policy = Policy.DQN
wandb_use = False

continous = False
if(current_policy == Policy.DDPG):
    continous = True

# Initialise the environment
if current_mode == MODE.TRAIN:
    # env = gym.make("CartPole-v1")
    env = gym.make("CarRacing-v2", continuous=continous)
elif current_mode == MODE.TEST:
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CarRacing-v2", render_mode="human", continuous=continous)

env = SkipFrame(env, skip=4)
env = gym_wrap.GrayScaleObservation(env)
env = gym_wrap.ResizeObservation(env, (84, 84))
env = gym_wrap.FrameStack(env, 4)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

obs, _ = env.reset()

timestep_n = 0
when2learn = 4 # in timesteps
when2sync = 5000 # in timesteps
when2save = 100000 # in timesteps

if current_mode == MODE.TRAIN:

    if wandb_use:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="my-awesome-project",

            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
            }
        )

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 2000
    else:
        num_episodes = 50

    if current_policy == Policy.DQN:
        manager = DQNManager(env)
    elif current_policy == Policy.DDPG:
        manager = DDPGManager(env)
    else:
        raise ValueError("Policy not supported")
    
    try:
        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = env.reset()
            manager.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)
            cumulated_reward = 0
            for t in count():
                timestep_n += 1
                action = manager.select_action(state)
                if(current_policy == Policy.DDPG):
                    observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                else:
                    observation, reward, terminated, truncated, _ = env.step(np.int64(action.item()))
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device)

                cumulated_reward += reward
                # Store the transition in memory
                reward = torch.tensor([reward], device=device)
                manager.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                if timestep_n % when2learn == 0:
                    manager.optimize_model()
                    manager.soft_update()

                # Perform one step of the optimization (on the policy network)
                
                if done:
                    manager.episode_durations.append(t + 1)
                    manager.rewards.append(cumulated_reward)
                    if(i_episode % 10 == 0):
                        print(f"Episode {i_episode} lasted {t + 1} steps")
                        stats = manager.get_stats()
                        print(f"Mean duration: {stats[0]}, Mean reward: {stats[1]}")

                    if wandb_use:
                        wandb.log({"episode_num": i_episode, "total_reward": cumulated_reward, "episode_duration": t+1})

                        if(current_policy == Policy.DDPG):
                            wandb.log({"mean_critic_loss": stats[2], "mean_actor_loss": stats[3]})
                    
                    # manager.hard_update()
                    break
    except KeyboardInterrupt:
        print("Training interrupted")

    print('Complete')
    manager.save_model()

    if wandb_use:
        wandb.finish()

elif current_mode == MODE.TEST:

    if current_policy == Policy.DQN:
        manager = DQNManager(env)
        # manager.policy_net.load_state_dict(torch.load("Models/Cartpole/model.pth", weights_only=True))
        manager.policy_net.load_state_dict(torch.load("Models/Racing/model.pth", weights_only=True))
        manager.policy_net.eval()
    elif current_policy == Policy.DDPG:
        manager = DDPGManager(env)
        # manager.actor_net.load_state_dict(torch.load("Models/Cartpole/model.pth", weights_only=True))
        manager.actor_net.load_state_dict(torch.load("actor.pth", weights_only=True))
        manager.actor_net.eval()

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    cumulated_reward = 0
    for t in count():
        # env.render()
        action = manager.select_greedy_action(state)
        print(action)
        if(current_policy == Policy.DDPG):
            observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        else:
            observation, reward, terminated, truncated, _ = env.step(np.int64(action.item()))
        done = terminated or truncated
        cumulated_reward += reward

        if done:
            break
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            # Move to the next state
            state = next_state

    print('Complete')
    print(f"Episode lasted {t + 1} steps")
    print(f"Total reward: {cumulated_reward}")

    env.close()