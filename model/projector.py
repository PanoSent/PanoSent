import torch.nn as nn

class MMInputProjector(nn.Module):
    def __init__(self, input_dim=1024, output_dim=4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)