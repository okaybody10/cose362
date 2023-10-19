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
    tensor = n * [0]
    tensor[k] = 1
    return tensor


class KernelSVM(nn.Module):
    # f should be a function from torch : number of feature -> torch : new space.
    def __init__(self, input_dim, num_classes, kernel=(lambda x: x), C=1.0, margin=1.0):
        super(KernelSVM, self).__init__()
        self.weight = nn.Parameter(torch.randn((input_dim, num_classes), requires_grad=True))
        self.bias = nn.Parameter(torch.randn((1, num_classes), requires_grad=True))
        self.C = C
        self.kernelToMatrix = lambda x: torch.stack([kernel(x[i]) for i in range(x.size()[0])])
        self.num_classes = num_classes
        self.margin = margin

    def train(self, X, y, epochs=100):
        kernel_x = self.kernelToMatrix(X)
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self(kernel_x, y)
            loss.backward()
            optimizer.step()

    def multiclass_hinge_loss_one_hot(self, outputs, y_one_hot):
        correct_class_scores = (outputs * y_one_hot).sum(1)
        margins = torch.clamp(outputs - correct_class_scores.unsqueeze(1) + self.margin, min=0)
        margins = margins * (1 - y_one_hot)  # Zero-ing for correct class
        loss = margins.sum(1).mean()
        return loss

    def forward(self, X, y):
        output = torch.matmul(X, self.weight) + self.bias
        hinge_loss = self.multiclass_hinge_loss_one_hot(output, y)
        reg_loss = 0.5 * self.weight.pow(2).sum()
        total_loss = reg_loss + self.C * hinge_loss
        return total_loss

    def predict_onehot(self, x):
        output = torch.matmul(self.kernelToMatrix(x), self.weight) + self.bias
        return torch.stack([_one_hot(self.num_classes, i) for i in torch.argmax(output, dim=1)])

    def predict_index(self, x):
        output = torch.matmul(self.kernelToMatrix(x), self.weight) + self.bias
        return torch.argmax(output, dim=1)

    def predict(self, x, using=predict_onehot):
        return using(self, x)