import os
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
model_base = "SVM"
traing_args = TrainingArguments(
    output_dir=f"./checkpoints/Bert_{model_base}_checkpoints/",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=12,
    warmup_ratio=0.1,
    warmup_steps=0,
    # **default_args
) # Optional

PATH = traing_args.output_dir

if not os.path.exists(PATH) :
    os.makedirs(PATH)

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

running_loss = 0.0
correct_label_wr, total_label_wr = 0, 0

writer = SummaryWriter('./runs/bert/' + model_base)

for epoch in range(traing_args.num_train_epochs) :
    model.train() 
    with tqdm(train_loader, unit="batch") as pbar:
        for i, data in enumerate(pbar) :
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch+1}")
            batch_train, batch_label = data['texts'], data['labels']
            with accelerator.accumulate(model) :
            # TODO: Seperate SVM & CRF
                now_loss, batch_labels, pad_output = model(batch_train, batch_label) # Output
                if model_base == "CRF" :
                    now_loss *= -1
                    running_loss += now_loss.mean()
                else :
                    running_loss += now_loss
                accelerator.backward(now_loss.mean())
                optimizer.step()
                scheduler.step()
            correct_label_wr += torch.sum((pad_output == batch_labels) & (batch_labels != 35) & (batch_labels != 32))
            total_label_wr += torch.sum((batch_labels != 35) & (batch_labels != 32))
            if i % 100 == 99 :
                writer.add_scalar('train/training_loss',
                                  running_loss / 1000,
                                  epoch * len(train_loader) + i + 1)
                running_loss = 0.0

            pbar.set_postfix(loss = now_loss.mean().item(), correct_label = correct_label_wr.item(), total_label = total_label_wr.item())
    # One epoch ends
    accuracy = correct_label_wr / total_label_wr
    writer.add_scalar('train/accuracy',
                        accuracy.item(),
                        epoch + 1)
    writer.add_scalar('train/correct labels',
                        correct_label_wr.item(),
                        epoch + 1)
    correct_label_wr, total_label_wr = 0, 0
    # Save state
    # accelerator.save_model(model, traing_args.output_dir)
    state = {
        'epoch' : epoch,
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'loss' : now_loss
    }
    torch.save(state, PATH + model_base + f"{epoch+1}_edit.pkl")
    # Validation Part
    print("==========Validation Start==========")
    correct_val, totals_val = 0, 0
    model.eval()
    with torch.no_grad() :
        for data in test_loader :
            batch_test, batch_test_label = data['texts'], data['labels']
            _, batch_val_labels, val_pad_output = model(batch_test, batch_test_label)
            correct_val += torch.sum((val_pad_output == batch_val_labels) & (batch_val_labels != 35) & (batch_val_labels != 32))
            totals_val += torch.sum((batch_val_labels != 35) & (batch_val_labels != 32))
    accuracy_val = correct_val / totals_val
    print(f"Correct labels: {correct_val.item()}, total labels: {totals_val.item()}, so accuracy is: {accuracy_val.item()}")
    writer.add_scalar('val/validation_accuracy',
                      accuracy_val.item(),
                      epoch + 1)
    writer.add_scalar('val/validation_labels',
                      correct_val.item(),
                      epoch + 1)