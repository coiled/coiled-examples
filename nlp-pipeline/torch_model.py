import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class HiddenLayerNet(nn.Module):
    def __init__(self, n_features=10, n_outputs=1, n_hidden=100, 
            n_extra_layers= 0, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        # need to add the layers via a module list of they won't be regitered
        self.extra_layers = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in
                range(n_extra_layers)])
        self.fc2 = nn.Linear(n_hidden, n_outputs)
        self.activation = getattr(F, activation)

    def forward(self, x, **kwargs):
        z = self.activation(self.fc1(x))
        for layer in self.extra_layers:
            z = self.activation(layer(z))
        return self.fc2(self.activation(z))



