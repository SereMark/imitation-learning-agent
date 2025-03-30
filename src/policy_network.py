import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions=5, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(6400, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.fc(self.features(x))