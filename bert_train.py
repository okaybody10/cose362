import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Tuple, Any
from accelerate import Accelerator
from transformers import AdamW, TrainingArguments, get_linear_schedule_with_warmup
from process import CustomDataset, collect_fn_bert, tagging, load_files
from bert_model import NERBertCRF, NERBertSVM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import get_parameter_names
from tqdm import tqdm

# Training
traing_args = TrainingArguments(
    output_dir="./checkpoints/Bert_CRF_checkpoints",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=12,
    warmup_ratio=0.1,
    warmup_steps=0,
    # **default_args
) # Optional

model_base = "CRF"
PATH = f'./checkpoints/Bert_{model_base}_checkpoints'

df = load_files() # Have to control path (argparser)
texts = df['form'].to_list()
ne = df['NE'].to_list()

train_texts, test_texts, train_ne, test_ne = train_test_split(texts, ne, test_size=0.2, random_state=42) # fix training dataset
train_dataset = CustomDataset(train_texts, train_ne)
test_dataset = CustomDataset(test_texts, test_ne)
train_loader = DataLoader(train_dataset, batch_size = traing_args.per_device_train_batch_size, shuffle=True, collate_fn=collect_fn_bert)
test_loader = DataLoader(test_dataset, batch_size = traing_args.per_device_eval_batch_size, shuffle=True, collate_fn=collect_fn_bert)

# Prepare Model, optimizer
if model_base == "CRF" :
    model = NERBertCRF()
else :
    model = NERBertSVM()

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": traing_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr= traing_args.learning_rate, eps= traing_args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, traing_args.warmup_steps, len(train_loader) * traing_args.num_train_epochs)
accelerator = Accelerator()
model, train_loader, test_loader, optimizer, scheduler = accelerator.prepare(model, train_loader, test_loader, optimizer, scheduler)

model.train() 
running_loss = 0.0

writer = SummaryWriter('./runs/' + model_base)

for epoch in range(traing_args.num_train_epochs) :
    with tqdm(train_loader, unit="batch") as pbar:
        for i, data in enumerate(pbar) :
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            batch_train, batch_label = data['texts'], data['labels']
            now_loss, output, batch_labels, pad_output = model(batch_train, batch_label) # Output
            now_loss *= -1
            running_loss += now_loss.mean()
            correct_label = torch.sum((pad_output == batch_labels) & (batch_labels != 35))
            accuracy = correct_label / torch.sum(batch_labels != 35)
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
            accelerator.backward(now_loss.mean())
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss = now_loss.mean().item(), accuracy = accuracy.item(), correct_label = correct_label.item())
    # One epoch ends
    state = {
        'epoch' : epoch,
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'loss' : now_loss
    }
    torch.save(state, PATH + model_base + f"{epoch}.pt")