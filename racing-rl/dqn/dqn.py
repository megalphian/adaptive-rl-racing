import torch.nn as nn
import torch.nn.functional as F

class DQN_simple(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_simple, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQN_CNN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_CNN, self).__init__()
        n_channels, width, height = n_observations
        self.layer1 = nn.Conv2d(n_channels, 6, kernel_size=5)
        self.layer2 = nn.Conv2d(6, 10, kernel_size=4, stride=2)
        self.layer3 = nn.Linear(10 * 9 * 9, 256)
        self.layer4 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.layer1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.layer2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 10 * 9 * 9)
        x = F.relu(self.layer3(x))
        return self.layer4(x)