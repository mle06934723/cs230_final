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
