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

def compute_features_deprecated(model, temploader):
    model.eval()
    trainFeatures = torch.rand(len(temploader.dataset), low_dim).t()
    with torch.no_grad():
        print("computing features")
        for batch_idx, (data, _, _) in tqdm(enumerate(temploader), total=len(temploader)):
            batchSize = data['input_ids'].size(0)
            _, features = model(data)
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
    return trainFeatures
    
def compute_features(model, temploader, batch_size=1):
    model.eval()
    all_feats = torch.rand(len(temploader.dataset), 768).t() # [dim, n]
    with torch.no_grad():
        for batch_idx, (text, _, _) in tqdm(enumerate(temploader), total=len(temploader)):
            features = model.encode(text, convert_to_tensor=True) #[1, dim]
            start_index = batch_idx * batch_size
            end_index = batch_idx * batch_size + batch_size
            all_feats[:, start_index:end_index] = features.data.t()
    return all_feats

def weighted_knn(temploader, features, noisy_labels, return_labels, bsz = 1):
    similarity_graph_all = torch.zeros(len(temploader.dataset), len(temploader.dataset))
    discrepancy_measure = torch.zeros((len(temploader.dataset.targets),))
    discrepancy_measure_pseudo_labels = torch.zeros((len(temploader.dataset.targets),))
    agreement_measure = torch.zeros((len(temploader.dataset.targets),))
    if return_labels:
        new_labels = torch.LongTensor(temploader.dataset.targets)
    else:
        new_labels = noisy_labels.clone()

    features = F.normalize(features, p=2, dim=0)

    with torch.no_grad():
        retrieval_one_hot_train = torch.zeros(k_val, num_classes)

        for batch_idx, (data, labels, index) in tqdm(enumerate(temploader), total=len(temploader)):
            # bsz = data['input_ids'].size(0)
            bsz = 1
            # make sure in dataset the train features are properly indexed
            features_transpose = features.t()[index]
            dist = torch.mm(features_transpose, features)
            if return_labels: # if in first loop against original noisy labels, compute similarity_graph_all
                similarity_graph_all[index] = dist.detach()
            # access diagonals of the matrix, or self
            dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  ## Self-contrast set to -1

            yd, yi = dist.topk(k_val, dim=1, largest=True, sorted=True)  ## Top-K similar scores and corresponding indexes
            candidates = new_labels.view(1, -1).expand(bsz, -1)  ## Replicate the labels per row to select
            retrieval = torch.gather(candidates, 1, yi)  ## Get top-K neighbour labels

            retrieval_one_hot_train.resize_(bsz * k_val, num_classes).zero_()
            retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = torch.exp(yd.clone().div_(sup_t))  ## Apply temperature to scores
            yd_transform[...] = 1.0  ## To avoid using similarities
            probs_corrected = torch.sum(
                torch.mul(retrieval_one_hot_train.view(bsz, -1, num_classes),
                yd_transform.view(bsz, -1, 1)), 1)
            probs_norm = probs_corrected / torch.sum(probs_corrected, dim=1)[:, None]
            prob_temp = probs_norm[torch.arange(0, bsz), labels]
            prob_temp = torch.clamp(prob_temp, min=1e-2, max=1 - 1e-2)
            discrepancy_measure[index] = -torch.log(prob_temp)

            if return_labels:
                sorted_pro, predictions_corrected = probs_norm.sort(1, True)
                targets_comparison = predictions_corrected[:, 0] # new_labels
            else:
                targets_comparison = noisy_labels[index]

            prob_temp = probs_norm[torch.arange(0, bsz), targets_comparison]
            prob_temp = torch.clamp(prob_temp, min=1e-2, max=1 - 1e-2)
            discrepancy_measure_pseudo_labels[index] = -torch.log(prob_temp)
            agreement_measure[index] = (torch.max(probs_norm, dim=1)[1] == labels).float().data

            if return_labels:
                new_labels[index] = targets_comparison

        if return_labels:
            return new_labels, similarity_graph_all

        else:
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
        samples_per_class = idx_class.sum()
        idx_class_indices = idx_class.nonzero(as_tuple=False).squeeze()
        discrepancy_class = final_discrepancy_measure[idx_class_indices]
        k_corrected = min(num_samples2select_class, samples_per_class)
        top_clean_class_relative_idx = torch.topk(discrepancy_class, k=int(k_corrected), largest=False, sorted=False).indices
        agreement_measure[idx_class_indices[top_clean_class_relative_idx]] = 1.0

    selected_examples = agreement_measure
    print('Selected examples:', torch.sum(selected_examples).item())
    return selected_examples

def map_similarity_to_confidence(similarity_scores, same_labels_mask, different_labels_mask):
    """
    This function takes similarity scores and maps them to confidence values.
    For same labels, the confidence is mapped closer to 1 as similarity increases.
    For different labels, the confidence is mapped closer to -1 as similarity decreases.
    """
    non_self_mask = ~torch.eye(similarity_scores.size(0), dtype=torch.bool, device=similarity_scores.device)

    # For **same labels**: high similarity corresponds to high positive confidence (close to 1)
    confidence_same = similarity_scores * 2 - 1  # Similarity [0, 1] -> Confidence [-1, 1]

    # For **different labels**: high similarity corresponds to high negative confidence (close to -1)
    confidence_diff = -(similarity_scores * 2 - 1)  # Similarity [0, 1] -> Confidence [1, -1]

    # Initialize a confidence matrix with zeros
    confidence = torch.zeros_like(similarity_scores)

    # Apply the confidence mapping for same labels
    confidence[same_labels_mask & non_self_mask] = confidence_same[same_labels_mask & non_self_mask]

    # Apply the confidence mapping for different labels
    confidence[different_labels_mask & non_self_mask] = confidence_diff[different_labels_mask & non_self_mask]

    return confidence

def select_pairs(selected_examples, similar_graph_all, noisy_labels, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        selected_examples = selected_examples.to(device).bool()  # Convert to boolean mask
        similarity_scores = similar_graph_all.to(device)
        noisy_labels = noisy_labels.to(device)

        total_num = len(noisy_labels)

        confidence_matrix = torch.zeros((total_num, total_num), dtype=torch.float, device=device)

        # Create masks for same and different labels
        same_labels_mask = torch.eq(noisy_labels.unsqueeze(0), noisy_labels.unsqueeze(1))  # True if labels match
        different_labels_mask = ~same_labels_mask

        # Create selected mask where both i and j are selected
        selected_mask = selected_examples.unsqueeze(0) & selected_examples.unsqueeze(1)

        # Map similarity scores to confidence values
        confidence_matrix = map_similarity_to_confidence(similarity_scores, same_labels_mask, different_labels_mask)

        # Compute the threshold value based on the quantile of similarities in the selected mask
        selected_similarity_scores = similarity_scores[selected_mask]
        selected_threshold = torch.quantile(selected_similarity_scores, beta)

        # Set low similarity pairs to 0 based on the threshold
        confidence_matrix[~selected_mask] = 0.0
        confidence_matrix[similarity_scores < selected_threshold] = 0.0

        # Count how many elements are set to 0 due to the selected_mask being False
        not_selected_mask = ~selected_mask
        count_not_selected = torch.sum(confidence_matrix[not_selected_mask] == 0)
        print(f"Examples not selected due to unconfident examples: {count_not_selected}")

        # Count how many elements are set to 0 due to similarity being below the threshold
        similarity_below_threshold_mask = similarity_scores < selected_threshold
        count_below_threshold = torch.sum(confidence_matrix[similarity_below_threshold_mask] == 0)
        print(f"Examples not selected due to being below threshold: {count_below_threshold}")

        # Make sure the matrix is symmetric (i, j) = (j, i)
        confidence_matrix = torch.triu(confidence_matrix) + torch.triu(confidence_matrix, diagonal=1).T

    return confidence_matrix.contiguous()

num_classes = 2
low_dim = 128
k_val = 50
alpha = 0.5
beta = 0.5
sup_t = 0.1

def pair_selection(model, train_features, trainloader):
    model.eval()
    temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=1, shuffle=False, num_workers=8)
    init_noisy_labels = torch.LongTensor(temploader.dataset.targets)
    train_new_labels, similarity_graph_all = weighted_knn(temploader, train_features, init_noisy_labels, True)
    agreement_measure, discrepancy_measure = weighted_knn(temploader, train_features, train_new_labels, False)
    selected_examples = select_examples(temploader, discrepancy_measure, agreement_measure)
    selected_pairs = select_pairs(selected_examples, similarity_graph_all, init_noisy_labels)
    return selected_examples, selected_pairs

features = compute_features_deprecated(model, train_dataloader)
examples, pairs = pair_selection(model, features, train_dataloader)
