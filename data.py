import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BinaryRewardModelDataset(Dataset):
    def __init__(self, chosen_texts, rejected_texts):
        self.texts = []
        self.targets = []

        # Flatten the data: add chosen texts with label 1 and rejected texts with label 0
        for chosen, rejected in zip(chosen_texts, rejected_texts):
            self.texts.append(chosen)  # Chosen text with label 1
            self.targets.append(1)  # Label 1 for chosen text
            self.texts.append(rejected)  # Rejected text with label 0
            self.targets.append(0)  # Label 0 for rejected text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get the text (either chosen or rejected) and label
        text = self.texts[idx]
        label = self.targets[idx]
        return text, label, idx  # Return the label as a tensor

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

