import torch
import torch.nn as nn
import numpy as np

class NN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, device=torch.device('cpu')) -> None:
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=3e-4)
        self.to(device)

    def check(self, x):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float32, device=self.device)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.check(x)
        return self.mlp(x)

    def update(x):
        pass

