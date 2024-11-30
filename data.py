import torch
import tqdm
from torch.utils.data import Dataset
from huggingface_hub import HfApi, Repository
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Value,
    ClassLabel
)
from utils import split_substring_data

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


def create_pretraining_dataset_schema(dataset, run_name): 
  all_train_examples = [] 
  for example in tqdm(dataset['train']):
    tups = split_substring_data(example)
    all_train_examples.extend(tups)
  
  all_test_examples = []
  for example in tqdm(dataset['test']):
    tups = split_substring_data(example)
    all_test_examples.extend(tups)
  
  train_data = all_train_examples
  test_data = all_test_examples

  features = Features({
      "sentence1": Value("string"),
      "sentence2": Value("string"),
      "score": ClassLabel(names=["0", "1"]),
  })
  train_dataset = Dataset.from_dict({
      "sentence1": [item[0] for item in train_data],
      "sentence2": [item[1] for item in train_data],
      "score": [item[2] for item in train_data]
  }, features=features)
  
  test_dataset = Dataset.from_dict({
      "sentence1": [item[0] for item in test_data],
      "sentence2": [item[1] for item in test_data],
      "score": [item[2] for item in test_data]
  }, features=features)
  
  split = train_dataset.train_test_split(test_size=0.2, seed=42)  # 20% for dev, 80% for train
  
  train_dataset = split['train']
  dev_dataset = split['test']

  dataset_dict = DatasetDict({
      "train": train_dataset,
      "dev": dev_dataset,
      "test": test_dataset
  })
  
  print(dataset_dict)
  dataset_dict.save_to_disk("split_contrastive_dataset")
  dataset_name = dataset_name  
  api = HfApi()
  repo_url = api.create_repo(dataset_name, exist_ok=True) 
  repo = Repository(local_dir=dataset_name, clone_from=repo_url)
  dataset_dict.push_to_hub(dataset_name)