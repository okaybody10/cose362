import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typing

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _one_hot(labels: torch.Tensor, num_classes) -> torch.LongTensor:
    return F.one_hot(labels, num_classes=num_classes) # One-hot vector
class KernelSVM(nn.Module):
    # f should be a function from torch : number of feature -> torch : new space.
    def __init__(self, input_dim=768, num_classes=34, kernel=(lambda x: x), C=1.0, margin=1.0):
        super(KernelSVM, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=True).to(device)
        self.C = C
        self.kernelToMatrix = lambda x: torch.stack([kernel(x[i]) for i in range(x.size()[0])])
        self.num_classes = num_classes
        self.margin = margin

    # def train(self, X, y, epochs=100):
    #     # kernel_x = self.kernelToMatrix(X)
    #     optimizer = optim.SGD(self.parameters(), lr=0.01)
    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
    #         loss = self(X, y)
    #         loss.backward()
    #         optimizer.step()

    def multiclass_hinge_loss_one_hot(self, outputs : torch.Tensor, y : torch.Tensor) -> torch.float :
        y = y.unsqueeze(-1).type(torch.LongTensor).to(device)
        # print(y)
        correct_class_scores = torch.gather(outputs, dim= -1, index= y).to(device) # (batch, seq, 1), gather ans value, also refer this link: https://pytorch.org/docs/stable/generated/torch.gather.html
        # print(correct_class_scores)
        margins = torch.clamp(outputs - correct_class_scores + self.margin, min=0).to(device) # Broadcasting, (bathc, seq, num_classes)
        # print(margins.size(), type(margins))
        # print(margins)
        margins = margins.scatter(-1, y, 0) # Refer this link: https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
        loss = torch.mean(torch.sum(margins, -1))
        # print(loss)
        return loss
        # correct_class_scores = (outputs * y_one_hot).sum(-1) # (batch, seq)
        # margins = margins * (1 - y_one_hot)  

    def forward(self, X, y):
        output = self.linear(X).to(device)
        hinge_loss = self.multiclass_hinge_loss_one_hot(output, y)
        print(hinge_loss)
        reg_loss = 0.5 * torch.norm(self.linear.weight, p=2)
        total_loss = hinge_loss + self.C * reg_loss # Fix regularization term
        return total_loss

    def predict_onehot(self, X):
        output = self.linear(X)
        return _one_hot(torch.argmax(output, dim=-1), self.num_classes) # output * num_classes
        # return torch.stack([_one_hot(self.num_classes, i) for i in torch.argmax(output, dim=-1)])

    def predict_index(self, X):
        output = self.linear(X).to(device)
        return torch.argmax(output, dim=-1)

    def predict(self, x, using=predict_onehot):
        return using(self, x)