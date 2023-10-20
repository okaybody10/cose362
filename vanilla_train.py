import os
import torch
from torch.utils.tensorboard import SummaryWriter
import metric
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from process import CustomDataset, collect_fn, load_files, tagging
from svm import KernelSVM

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH = './checkpoints/'

df = load_files() # Have to control path (argparser)
texts = df['form'].to_list()
ne = df['NE'].to_list()

# tokenizer
train_texts, test_texts, train_ne, test_ne = train_test_split(texts, ne, test_size=0.2, random_state=42) # fix training dataset
train_dataset = CustomDataset(train_texts, train_ne)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, collate_fn=collect_fn)

model_type = 'SVM' # Option: SVM, CRF & argparser
if model_type == 'SVM' :
    model = KernelSVM().to(device)
else :
    pass # Later, it will be replaced by CRF()

writer = SummaryWriter('./runs/' + model_type)

model.train()
epochs = 10

running_loss = 0.0

for epoch in range(epochs) :
    with tqdm(train_loader, unit="batch") as pbar:
        for i, data in enumerate(pbar) :
            pbar.set_description(f"Epoch {epoch}")
            batch_train, batch_label = data['texts'].to(device), data['labels'].to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()

            model.eval()
            with torch.no_grad() : 
                output = model.predict_index(batch_train).to(device)
                correct_label = torch.sum((output == batch_label) & (batch_label != 33) & (batch_label))
                accuracy = correct_label / torch.sum(batch_label != 33)
            model.train()

            loss = model(batch_train, batch_label)
            running_loss += loss.item()

            if i % 100 == 99 :
                writer.add_scalar('training_loss',
                                  running_loss / 1000,
                                  epoch * len(train_loader) + i + 1)
                running_loss = 0.0
                writer.add_scalar('accuracy',
                                  accuracy.item(),
                                  epoch * len(train_loader) + i + 1)
                writer.add_scalar('correct labels',
                                  correct_label.item(),
                                  epoch * len(train_loader) + i + 1)

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss = loss.item(), accuracy = accuracy.item(), correct_label = correct_label.item())
    # One epoch ends
    state = {
        'epoch' : epoch,
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'loss' : loss
    }
    torch.save(state, PATH + model_type + f"{epoch}.pt")