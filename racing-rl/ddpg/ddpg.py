import torch.nn as nn
import torch.nn.functional as F
    
class Actor_CNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Actor_CNN, self).__init__()
        n_channels, width, height = n_observations
        latent_dim = 10*9*9
        self.encoder_cnn = nn.ModuleList([
            nn.Conv2d(n_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 10, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.linear = nn.ModuleList([
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        ])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x.shape)
        for layer in self.encoder_cnn:
            x = layer(x)
        x = x.view(-1, 10 * 9 * 9)
        for layer in self.linear:
            x = layer(x)
        return x
    
class Critic_CNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Critic_CNN, self).__init__()
        n_channels, width, height = n_observations
        latent_dim = 10*9*9
        self.encoder_cnn = nn.ModuleList([
            nn.Conv2d(n_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 10, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.linear = nn.ModuleList([
            nn.Linear(latent_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        ])

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x.shape)
        for layer in self.encoder_cnn:
            x = layer(x)
        x = x.view(-1, 10 * 9 * 9)
        for layer in self.linear:
            x = layer(x)
        return x