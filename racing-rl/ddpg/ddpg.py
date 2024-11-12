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
        self.latent_dim = 32*9*9
        self.encoder_cnn = nn.ModuleList([
            nn.Conv2d(n_channels, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.linear = nn.ModuleList([
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        ])

        self.min_action = torch.tensor([-1, 0, 0]).to(device)
        self.max_action = torch.tensor([1, 1, 1]).to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x.shape)
        for layer in self.encoder_cnn:
            x = layer(x)
        x = x.view(-1, self.latent_dim)
        for layer in self.linear:
            x = layer(x)
        x = torch.clamp(x, self.min_action, self.max_action)
        return x
    
class Critic_CNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Critic_CNN, self).__init__()
        n_channels, width, height = n_observations
        self.latent_dim = 32*9*9
        self.encoder_cnn = nn.ModuleList([
            nn.Conv2d(n_channels, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.linear = nn.ModuleList([
            nn.Linear(self.latent_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ])

    def forward(self, x, u):
        for layer in self.encoder_cnn:
            x = layer(x)
        x = x.view(-1, self.latent_dim)
        x = torch.cat([x, u], 1)
        for layer in self.linear:
            x = layer(x)
        return x