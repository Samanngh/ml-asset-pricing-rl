import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, n_inputs, n_neurons, n_actions):
        """
        n_inputs   : state dimension (lookback * n_assets + n_assets)
        n_neurons  : list of hidden layer sizes, e.g. [128, 64, 32]
        n_actions  : number of assets (output dimension)
        """
        super().__init__()
        self.h1 = nn.Linear(n_inputs, n_neurons[0])
        self.h2 = nn.Linear(n_neurons[0], n_neurons[1])
        self.h3 = nn.Linear(n_neurons[1], n_neurons[2])
        self.out = nn.Linear(n_neurons[2], n_actions)

    def forward(self, state):
        """
        Forward pass:
        - state: tensor of shape (batch_size, n_inputs)
        - returns: portfolio weights of shape (batch_size, n_actions)
        """
        x = F.relu(self.h1(state))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        weights = F.softmax(self.out(x), dim=1)  # ensures sum=1
        return weights

    def save(self, model_path):
        """Save model parameters to a checkpoint."""
        checkpoint = {"model_state_dict": self.state_dict()}
        torch.save(checkpoint, model_path)

    def load(self, model_path):
        """Load model parameters from a checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        return self
