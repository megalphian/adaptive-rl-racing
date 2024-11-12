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
        self.latent_dim = 10*9*9
        
        self.cnn1 = nn.Conv2d(n_channels, 5, kernel_size=5,)
        self.cnn2 = nn.Conv2d(5, 10, kernel_size=4, stride=2)
        
        self.fc1 = nn.Linear(self.latent_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.cnns = [self.cnn1, self.cnn2]
        self.linears = [self.fc1, self.fc2]

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.cnns:
            x = layer(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.latent_dim)
        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)
        x = F.tanh(x)
        return x
    
class Critic_CNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Critic_CNN, self).__init__()
        n_channels, width, height = n_observations
        self.latent_dim = 5*7*7
        
        self.cnn1 = nn.Conv2d(n_channels, 5, kernel_size=5)
        self.cnn2 = nn.Conv2d(5, 10, kernel_size=4)
        self.cnn3 = nn.Conv2d(10, 5, kernel_size=4)

        self.cnns = [self.cnn1, self.cnn2, self.cnn3]

        self.fc1 = nn.Linear(self.latent_dim + n_actions, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.linears = [self.fc1, self.fc2, self.fc3]
        
    def forward(self, x, u):
        for layer in self.cnns:
            x = layer(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.latent_dim)
        x = torch.cat([x, u], 1)
        for layer in self.linears:
            x = layer(x)
            if layer != self.linears[-1]:
                x = F.relu(x)
        return x