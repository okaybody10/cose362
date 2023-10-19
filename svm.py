import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim


def _one_hot(n, k):
    tensor = torch.zeros(n)
    tensor[k] = 1
    return tensor

def _one_hot_list(n, k):
    # Create a tensor of zeros of size n
    tensor = n * [0]

    # Set the k-th dimension to 1
    tensor[k] = 1

    return tensor



class KernelSVM(nn.Module):
    # f should be a function from torch : number of feature -> torch : new space.
    def __init__(self, input_dim, num_classes, kernel=(lambda x: x), C=1.0):
        super(KernelSVM, self).__init__()
        self.weight = nn.Parameter(torch.randn((input_dim, num_classes), requires_grad=True))
        self.bias = nn.Parameter(torch.randn((1, num_classes), requires_grad=True))
        self.C = C
        self.kernelToMatrix = lambda x : torch.stack([kernel(x[i]) for i in range(x.size()[0])])
        self.num_classes = num_classes
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def train(self, X, y, epochs=100):
        kernel_X = self.kernelToMatrix(X)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            # binary_labels ∈ {+1, -1}
            loss = self(kernel_X, y)
            loss.backward()
            self.optimizer.step()

    def forward(self, X, y): ## 여기서는 kernel을 적용하지 않음 - 이미 적용되어있음
        output = torch.matmul(X, self.weight) + self.bias
        # Hinge loss
        loss = 0.5 * self.weight.pow(2).sum() + self.C * torch.clamp(1 - y * output, min=0).sum()
        return loss

    def predict(self, x):
        output = torch.matmul(self.kernelToMatrix(x), self.weight) + self.bias
        return torch.stack([_one_hot(self.num_classes, i) for i in torch.argmax(output, dim=1)])
