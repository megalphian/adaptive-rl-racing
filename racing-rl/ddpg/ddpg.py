import torch.nn as nn
import torch.nn.functional as F

import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_observations, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
        
class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_observations, 512)
        self.fc2 = nn.Linear(512 + n_actions, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, x, u):
        x = F.relu(self.fc1(x))
        x = torch.cat([x, u], 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor_CNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Actor_CNN, self).__init__()
        n_channels, width, height = n_observations
        self.n_actions = n_actions
        self.latent_dim = 10368
        
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fcs = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            # add normalization layer
            nn.BatchNorm1d(128),
            nn.Linear(128, n_actions),
            nn.BatchNorm1d(n_actions),
            nn.Tanh()
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.encoder(x)
        x = self.fcs(x)
        return x
    
class Critic_CNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Critic_CNN, self).__init__()
        n_channels, width, height = n_observations
        self.latent_dim = 3240
        
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 5, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=4),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fcs = nn.Sequential(
            nn.Linear(self.latent_dim + n_actions, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, u):
        x = self.encoder(x)
        x = torch.cat([x, u], 1)
        x = self.fcs(x)
        return x