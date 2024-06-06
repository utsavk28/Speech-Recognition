import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Softmax()
        )

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred

    def print_trainable_params(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)
        print(f"{'='*50} Feed Forward Neural Network {'='*50}")
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)
        return total_params, trainable_params
