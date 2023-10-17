import torch
import torch.nn as nn
import torch.optim as optim


class BinarySVM(nn.Module):
    def __init__(self, input_dim, C=1.0):
        super(BinarySVM, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
        self.C = C

    def forward(self, x, y):
        output = torch.matmul(x, self.weight) + self.bias
        # Hinge loss
        loss = 0.5 * self.weight.pow(2).sum() + self.C * torch.clamp(1 - y * output, min=0).sum()
        return loss

    def predict(self, x):
        output = torch.matmul(x, self.weight) + self.bias
        return output


class MultiClassSVM:
    def __init__(self, input_dim, num_classes, C=1.0):
        self.svms = [BinarySVM(input_dim, C) for _ in range(num_classes)]
        self.optimizers = [optim.SGD(svm.parameters(), lr=0.01) for svm in self.svms]

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for i, (svm, optimizer) in enumerate(zip(self.svms, self.optimizers)):
                optimizer.zero_grad()
                binary_labels = (y == i).float() * 2 - 1  # Convert to +1/-1 labels
                loss = svm(X, binary_labels)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        outputs = [svm.predict(X) for svm in self.svms]
        outputs = torch.stack(outputs, dim=1)  # Shape: [num_samples, num_classes]
        return torch.argmax(outputs, dim=1)
