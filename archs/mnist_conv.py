from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def Conv4FC3():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    return model

