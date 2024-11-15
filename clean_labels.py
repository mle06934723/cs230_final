# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from tqdm import tqdm
from typing import Dict

class IntegrityClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes, hidden_size, low_dim):
        super(IntegrityClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dim_reducer = nn.Linear(hidden_size, low_dim)

    def forward(self, inputs: Dict[str, torch.Tensor], feat_classifier=False):
        out = self.encoder(**inputs)
        out_ = out.pooler_output
        if(feat_classifier):
            return out_
        outContrast = self.dim_reducer(out_)
        outPred = self.classifier(out_)
        return outPred, outContrast

encoder = AutoModel.from_pretrained('xlm-roberta-large')

model = IntegrityClassificationModel(encoder, 2, 1024, 128)

class AnthropicDataset(Dataset):
    def __init__(self, dataset, tokenizer, num_samples=1000, partition='train'):
        self.tokenizer = tokenizer
        self.dataset = dataset
        mini = dataset[partition].shuffle(seed=42).select(range(num_samples))
        tokenized = mini.map(self.tokenize_function)
        examples = []
        self.num_class = 2
        for elem in tokenized:
            chosen = elem['chosen']
            rejected = elem['rejected']
            chosen_tokens = elem['chosen_tokens_and_mask']
            rejected_tokens = elem['rejected_tokens_and_mask']
            chosen_example = {'input_ids': chosen_tokens['input_ids'], 'attention_mask': chosen_tokens['attention_mask'], 'labels': [1]}
            rejected_example = {'input_ids': rejected_tokens['input_ids'], 'attention_mask': rejected_tokens['attention_mask'], 'labels': [0]}
            # Add the new examples to the transformed dataset
            examples.extend([chosen_example, rejected_example])
        self.examples = pd.DataFrame(examples)
        self.exs = []
        self.targets = []
        for idx in range(len(self.examples)):
            labels = self.examples.iloc[idx]['labels']
            self.targets.append(labels)

    def tokenize_function(self, examples):
        chosen_tokens_and_mask = self.tokenizer(examples['chosen'], padding="max_length", truncation=True, return_tensors='pt')
        rejected_tokens_and_mask = self.tokenizer(examples['rejected'], padding="max_length", truncation=True, return_tensors='pt')
        return {
            'chosen_tokens_and_mask': chosen_tokens_and_mask,
            'rejected_tokens_and_mask': rejected_tokens_and_mask,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        data = {
            "input_ids": torch.squeeze(torch.tensor(self.examples.iloc[idx]['input_ids'])),
            "attention_mask": torch.squeeze(torch.tensor(self.examples.iloc[idx]['attention_mask'])),

        }
        labels = torch.tensor(self.examples.iloc[idx]['labels'])
        return data, labels, idx


dataset = load_dataset("Anthropic/hh-rlhf")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

train_dataset = AnthropicDataset(dataset, tokenizer, num_samples=50, partition='train')
eval_dataset = AnthropicDataset(dataset, tokenizer, num_samples=1000, partition='test')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

num_classes = 2
low_dim = 128
k_val = 10
alpha = 0.5
beta = 0.25
sup_t = 0.1

def feature_compute(model, temploader):
    trainFeatures = torch.rand(len(temploader.dataset), low_dim).t()
    with torch.no_grad():
        print("computing features")
        for batch_idx, (data, _, _) in tqdm(enumerate(temploader), total=len(temploader)):
            batchSize = data['input_ids'].size(0)
            _, features = model(data)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
    return trainFeatures

def weighted_knn(temploader, trainFeatures, noisy_labels, return_labels):
    similarity_graph_all = torch.zeros(len(temploader.dataset), len(temploader.dataset))
    discrepancy_measure = torch.zeros((len(temploader.dataset.targets),))
    discrepancy_measure_pseudo_labels = torch.zeros((len(temploader.dataset.targets),))

    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, num_classes)

        for batch_idx, (data, labels, index) in tqdm(enumerate(temploader), total=len(temploader)):
            batchSize = data['input_ids'].size(0)
            features_transpose = trainFeatures.t()[index]
            dist = torch.mm(features_transpose, trainFeatures)
            if return_labels:
                similarity_graph_all[index] = dist.detach()
            dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  ## Self-contrast set to -1

            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  ## Top-K similar scores and corresponding indexes
            candidates = noisy_labels.view(1, -1).expand(batchSize, -1)  ## Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi)  ## Get top-K neighbour labels

            retrieval_one_hot_train.resize_(batchSize * k_val, num_classes).zero_()
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(yd.clone().div_(sup_t))  ## Apply temperature to scores
            yd_transform[...] = 1.0  ## To avoid using similarities
            probs_corrected = torch.sum(
                torch.mul(retrieval_one_hot_train.view(batchSize, -1, num_classes),
                yd_transform.view(batchSize, -1, 1)), 1)
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]

            prob_temp = probs_norm[torch.arange(0, batchSize), labels]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure[index] = -torch.log(prob_temp)

            if return_labels:
                sorted_pro, predictions_corrected = probs_norm.sort(1, True)
                label_comparison = predictions_corrected[:, 0]
            else:
                label_comparison = noisy_labels[index]

            prob_temp = probs_norm[torch.arange(0, batchSize), label_comparison]
            prob_temp[prob_temp <= 1e-2] = 1e-2
            prob_temp[prob_temp > (1 - 1e-2)] = 1 - 1e-2
            discrepancy_measure_pseudo_labels[index] = -torch.log(prob_temp)

            if return_labels:
                labels = noisy_labels.clone()
                labels[index] = label_comparison
                return labels, similarity_graph_all

            else:
                agreement_measure = torch.zeros((len(temploader.dataset.targets),))
                agreement_measure[index.data] = (torch.max(probs_norm, dim=1)[1] == labels).float().data
                return agreement_measure, discrepancy_measure


def select_examples(temploader, final_discrepancy_measure, agreement_measure):
    num_clean_per_class = torch.zeros(num_classes)
    targets = torch.tensor(temploader.dataset.targets).squeeze()  # Shape: [N] (1D tensor of class labels)

    # Step 1: Count clean examples for each class
    for i in range(num_classes):
        idx_class = targets == i
        num_clean_per_class[i] = torch.sum(agreement_measure[idx_class])

    # Step 2: Calculate median number of clean examples per class
    num_samples2select_class = torch.median(num_clean_per_class.float()).item()

    # Reset the agreement measure to zero
    agreement_measure = torch.zeros((len(targets),))

    # Step 3: Select examples for each class
    for i in range(num_classes):
        idx_class = targets == i
        samplesPerClass = idx_class.sum()
        idx_class_indices = idx_class.nonzero(as_tuple=False).squeeze()
        discrepancy_class = final_discrepancy_measure[idx_class_indices]
        k_corrected = min(num_samples2select_class, samplesPerClass)
        top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False).indices
        agreement_measure[idx_class_indices[top_clean_class_relative_idx]] = 1.0

    selected_examples = agreement_measure
    print('Selected examples:', torch.sum(selected_examples).item())

    return selected_examples

def select_pairs(selected_examples, similar_graph_all, noisy_labels):
    with torch.no_grad():
        index_selected = torch.nonzero(selected_examples, as_tuple=True)[0].cpu()
        total_selected_num = len(index_selected)
        total_num = len(noisy_labels)

        noisy_pairs = torch.eq(noisy_labels, noisy_labels.t())  # Shape: [total_num, total_num]
        selected_pairs = noisy_pairs[index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num),
                                      index_selected.unsqueeze(0).expand(total_selected_num, total_selected_num)].clone()

        temp_graph = similar_graph_all[index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num),
                                       index_selected.unsqueeze(0).expand(total_selected_num, total_selected_num)]

        selected_threshold = torch.quantile(temp_graph[selected_pairs], beta)
        print('selected_threshold:', selected_threshold)

        temp = torch.zeros(total_num, total_num, dtype=torch.bool)
        noisy_pairs = torch.where(similar_graph_all < selected_threshold, temp, noisy_pairs)
        noisy_pairs[index_selected.unsqueeze(1).expand(total_selected_num, total_selected_num),
                    index_selected.unsqueeze(0).expand(total_selected_num, total_selected_num)] = selected_pairs

        # Return the final selected pairs
        final_selected_pairs = noisy_pairs

    return final_selected_pairs.contiguous()

def pair_selection_no_device(model, train_features, trainloader):
    model.eval()
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=1, shuffle=False, num_workers=8)
    init_noisy_labels = torch.LongTensor(temploader.dataset.targets)
    train_new_labels, similarity_graph_all = weighted_knn(temploader, train_features, init_noisy_labels, True)
    agreement_measure, discrepancy_measure = weighted_knn(temploader, train_features, train_new_labels, False)
    agreement_measure = torch.randint(0, 2, (100,))
    selected_examples = select_examples(temploader, discrepancy_measure, agreement_measure)
    selected_pairs = select_pairs(selected_examples, similarity_graph_all, init_noisy_labels)
    return selected_examples, selected_pairs

features = feature_compute(model, train_dataloader)
examples, pairs = pair_selection_no_device(model, features, train_dataloader)