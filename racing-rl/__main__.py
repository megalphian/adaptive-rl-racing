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
from core.envs import EnvMode

from enum import Enum

import wandb

class MODE(Enum): 
    TRAIN = 0
    TEST = 1

class Policy(Enum):
    DQN = 0
    DDPG = 1

current_mode = MODE.TRAIN
current_policy = Policy.DDPG
current_env = EnvMode.RACING
wandb_use = True

preload_model = True

continous = False
if(current_policy == Policy.DDPG):
    continous = True

# Initialise the environment
if current_mode == MODE.TRAIN:
    if wandb_use:
        if current_env == EnvMode.RACING:
            env = gym.make("CarRacing-v3", continuous=continous)
        elif current_env == EnvMode.PENDULUM:
            env = gym.make("Pendulum-v1", g=9.81)
    else:
        if current_env == EnvMode.RACING:
            env = gym.make("CarRacing-v3", continuous=continous, render_mode="human")
        elif current_env == EnvMode.PENDULUM:
            env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

elif current_mode == MODE.TEST:
    if current_env == EnvMode.RACING:
        env = gym.make("CarRacing-v3", render_mode="human", continuous=continous)
    elif current_env == EnvMode.PENDULUM:
        env = gym.make("Pendulum-v1", render_mode="human", g=9.81)


if current_env == EnvMode.RACING:
    env = SkipFrame(env, skip=4)
    env = gym_wrap.GrayscaleObservation(env)
    env = gym_wrap.ResizeObservation(env, (84, 84))
    env = gym_wrap.FrameStackObservation(env, stack_size=4)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
cpu_device = torch.device("cpu")

obs, _ = env.reset()

timestep_n = 0
when2learn = 2 # in timesteps
when2sync = 5000 # in timesteps
when2save = 10000 # in timesteps

if current_mode == MODE.TRAIN:

    if wandb_use:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="adaptive-rl",
            name="DDPG-Test-Racing",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.001,
            "architecture": "DDPG",
            "dataset": "CIFAR-100",
            "epochs": 10,
            },
        )

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 2000
    else:
        num_episodes = 50

    if current_policy == Policy.DQN:
        manager = DQNManager(env)
    elif current_policy == Policy.DDPG:
        manager = DDPGManager(env, current_env, wandb_use)
        if preload_model:
            manager.actor_net.load_state_dict(torch.load("Models/Racing/DDPG-v2/actor.pth", weights_only=True))
            manager.critic_net.load_state_dict(torch.load("Models/Racing/DDPG-v2/critic.pth", weights_only=True))
    else:
        raise ValueError("Policy not supported")
    
    try:
        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=cpu_device)
            cumulated_reward = 0
            manager.noise_generator.reset()
            for t in count():
                timestep_n += 1
                action = manager.select_action(state)
                if(current_policy == Policy.DDPG):
                    observation, reward, terminated, truncated, _ = env.step(action.numpy())
                else:
                    observation, reward, terminated, truncated, _ = env.step(np.int64(action.item()))
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=cpu_device)

                cumulated_reward += reward
                # Store the transition in memory
                reward = torch.tensor([reward], device=cpu_device)
                manager.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                manager.optimize_model()

                if timestep_n % when2learn == 0:
                    manager.soft_update()

                if timestep_n % when2save == 0:
                    manager.save_model()

                #### Perform one step of the optimization (on the policy network)
                
                if done:
                    manager.episode_durations.append(t + 1)
                    manager.rewards.append(cumulated_reward)
                    stats = manager.get_stats()
                    if(i_episode % 10 == 0):
                        print(f"Episode {i_episode} lasted {t + 1} steps")
                        print(f"Mean duration: {stats[0]}, Mean reward: {stats[1]}")

                    if wandb_use:
                        results_dict = {"episode_num": i_episode, "total_reward": cumulated_reward, "episode_duration": t+1, 'mean_reward': stats[1]}
                        
                        wandb.log(results_dict)
                    
                    break
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        env.close()

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
        manager = DDPGManager(env, current_env, False)
        # manager.actor_net.load_state_dict(torch.load("Models/Cartpole/model.pth", weights_only=True))
        manager.actor_net.load_state_dict(torch.load("Models/Racing/DDPG-v2/actor.pth", weights_only=True))
        # manager.actor_net.load_state_dict(torch.load("actor.pth", weights_only=True))
        manager.actor_net.eval()

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=cpu_device)
    cumulated_reward = 0
    for t in count():
        # env.render()
        
        if(current_policy == Policy.DDPG):
            action = manager.select_greedy_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.numpy())
        else:
            action = manager.select_greedy_action(state)
            observation, reward, terminated, truncated, _ = env.step(np.int64(action.item()))
        done = terminated or truncated
        cumulated_reward += reward

        if done:
            break
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=cpu_device)
            # Move to the next state
            state = next_state

    print('Complete')
    print(f"Episode lasted {t + 1} steps")
    print(f"Total reward: {cumulated_reward}")

    env.close()