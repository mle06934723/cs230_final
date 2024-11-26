# -*- coding: utf-8 -*-
"""Train xlm-r playground

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QDFalMeHgnIzmi2I0ATgMV4eiw0--Fhw
"""

pip install datasets transformers evaluate

import pandas as pd
from datasets import Dataset
import evaluate
from datasets import load_dataset
import torch
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

scaler = GradScaler()

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)

num_classes = 2

model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=num_classes)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

for param in model.roberta.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")


def custom_tokenize_and_pad(sentence, max_length=512):
    tokens = tokenizer.tokenize(normalize_text(sentence))
    if len(tokens) > max_length:
        tokens = tokens[-max_length:]  
    padding_length = max_length - len(tokens)
    if padding_length > 0:
        tokens = [tokenizer.pad_token] * padding_length + tokens 
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    attention_mask = [1] * (max_length - padding_length) 
    attention_mask += [0] * padding_length  
    input_ids = torch.tensor(token_ids)
    attention_mask_tensor = torch.tensor(attention_mask)

    assert len(input_ids) == max_length
    assert len(attention_mask_tensor) == max_length
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask_tensor
    }

train_ds = load_dataset("Anthropic/hh-rlhf", split='train')
test_ds = load_dataset("Anthropic/hh-rlhf", split='test[0:2000]')
def tokenize_function(examples):
    chosen_tokens = custom_tokenize_and_pad(examples['chosen'])
    rejected_tokens = custom_tokenize_and_pad(examples['rejected'])
    return {'chosen_tokens': chosen_tokens, 'rejected_tokens': rejected_tokens}

tokenized_train = train_ds.map(tokenize_function)
tokenized_test = test_ds.map(tokenize_function)

all_examples = []

for elem in tqdm(tokenized_train):
  chosen = elem['chosen']
  rejected = elem['rejected']
  chosen_tokens = elem['chosen_tokens']
  rejected_tokens = elem['rejected_tokens']
  chosen_example = {'input_ids': chosen_tokens['input_ids'], 'attention_mask': chosen_tokens['attention_mask'], 'labels': [1]}
  rejected_example = {'input_ids': rejected_tokens['input_ids'], 'attention_mask': rejected_tokens['attention_mask'], 'labels': [0]}
  # Add the new examples to the transformed dataset
  all_examples.extend([chosen_example, rejected_example])

train_ex = pd.DataFrame(all_examples)

train_ex

small_eval = []

for elem in tqdm(tokenized_test):
  chosen = elem['chosen']
  rejected = elem['rejected']
  chosen_tokens = elem['chosen_tokens']
  rejected_tokens = elem['rejected_tokens']
  chosen_example = {'input_ids': chosen_tokens['input_ids'], 'attention_mask': chosen_tokens['attention_mask'], 'labels': [1]}
  rejected_example = {'input_ids': rejected_tokens['input_ids'], 'attention_mask': rejected_tokens['attention_mask'], 'labels': [0]}
  # Add the new examples to the transformed dataset
  small_eval.extend([chosen_example, rejected_example])

eval_ex = pd.DataFrame(small_eval)

eval_ex

from torch.utils.data import Dataset
class AnthropicDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.examples.iloc[idx]['input_ids']),
            "attention_mask": torch.tensor(self.examples.iloc[idx]['attention_mask']),
            "labels": torch.tensor(self.examples.iloc[idx]['labels'])
        }

train_dataset = AnthropicDataset(train_ex)
eval_dataset = AnthropicDataset(eval_ex)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=16,
    worker_init_fn=worker_init_fn,
    num_workers=8,
    pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=True,
    batch_size=8,
    worker_init_fn=worker_init_fn,
    num_workers=8,
    pin_memory=True
)

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()

def test(epoch, model):
  metric = evaluate.load("accuracy")
  progress_bar = tqdm(range(len(eval_dataloader)))
  model.eval()
  running_loss = 0
  with torch.no_grad():
    for batch in eval_dataloader:
        batch = {
                "input_ids": batch["input_ids"].squeeze(1).to(device=device, non_blocking=True),
                "attention_mask": batch["attention_mask"].squeeze(1).to(device=device, non_blocking=True),
                "labels": batch["labels"].to(device=device, non_blocking=True),
        }
        outputs = model(**batch)
        loss = outputs.loss
        running_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)

  test_loss=running_loss/len(eval_dataloader)
  print(f"Epoch {epoch+1}/ - Avg Test Loss: {test_loss:.4f}")
  metric.compute()

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    total_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        batch = {
            "input_ids": batch["input_ids"].squeeze(1).to(device=device, non_blocking=True),
            "attention_mask": batch["attention_mask"].squeeze(1).to(device=device, non_blocking=True),
            "labels": batch["labels"].to(device=device, non_blocking=True),
        }
        with autocast():
          outputs = model(**batch)
          loss = outputs.loss

        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        progress_bar.update(1)

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    test(epoch, model)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }
