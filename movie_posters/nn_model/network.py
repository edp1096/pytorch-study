import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 10),
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 25),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(25, 1),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((7, 7))(
                nn.Linear(1, 1),
                nn.ReLU(),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(1, 1, bias=True),
                nn.ReLU(),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(1, 1, bias=True),
            ),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
