from collections import defaultdict
from transformers import Trainer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm 
import torch.nn as nn 
from accelerate.utils import gather_object
import torch.nn.functional as F
from utils import pair_selection 
from typing import List, Optional, Tuple, Union, Any, Dict, Iterable
from data import CurriculumContrastiveRewardTrainDataset 
import wandb 
from transformers.trainer_pt_utils import nested_detach
from sentence_transformers import SentenceTransformer 
from trl.trainer.utils import (
    decode_and_strip_padding, print_rich_table
)
import pandas as pd 

class SentenceTransformerRewardDiscriminator(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.5):
        super(SentenceTransformerRewardDiscriminator, self).__init__()
        model = SentenceTransformer(model_name)
        self.encoder = model 
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.encoder.get_sentence_embedding_dimension(), num_classes)

    def forward(self, text) -> Dict[str, torch.Tensor]: 
        embeddings = self.encoder.encode(text, convert_to_tensor=True)
        logits = self.classifier(self.dropout(embeddings))
        return {
            "sentence_embedding": embeddings,
            "logits": logits
        }

class ContrastiveRewardTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.reward_weight = 0.9 
        self.margin = 0.5 

    def compute_features(self, dataloader):
        all_feats = torch.rand(len(dataloader.dataset), self.model.encoder.get_sentence_embedding_dimension()).t() # [dim, n]
        with torch.no_grad():
            for batch_idx, (text, _, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
                features = self.model.encoder.encode(text, convert_to_tensor=True) #[1, dim]
                start_index = batch_idx * self.args.per_device_train_batch_size
                end_index = batch_idx * self.args.per_device_train_batch_size + self.args.per_device_train_batch_size
                all_feats[:, start_index:end_index] = features.data.t()
        return all_feats

    def get_train_dataloader(self):
        # init_data = BinaryRewardModelDataset(positives, negatives) pass this in as train_dataset
        init_dataloader = DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=False)
        self.model.eval() 
        features = self.compute_features(init_dataloader)
        selected_dataset = CurriculumContrastiveRewardTrainDataset(self.train_dataset, features, init_dataloader)
        # Create a DataLoader with the subset of data
        self.model.train() 
        return DataLoader(selected_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=False)

    def get_eval_dataloader(self, eval_dataset):
        polarity = [0] * len(eval_dataset)
        eval_dataset = eval_dataset.add_column("polarity", polarity)
        return DataLoader(eval_dataset, batch_size=self.args.per_device_eval_batch_size, shuffle=True)

    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            _, logits, _ = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            chosen_text = inputs["chosen"]
            rejected_text = inputs["rejected"]
            table["chosen_text"].extend(gather_object(chosen_text))
            table["rejected_text"].extend(gather_object(rejected_text))
            table["logits"].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])
            )
            if num_print_samples >= 0 and len(table["chosen_text"]) >= num_print_samples:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})
    
    def evaluate(self, *args, **kwargs):
        num_print_samples = kwargs.pop("num_print_samples", 4)
        self.visualize_samples(num_print_samples)
        return super().evaluate(*args, **kwargs)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = (inputs["chosen"], inputs["rejected"], inputs["polarity"])
        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits1 = logits_dict["rewards_sentence1"]
        logits2 = logits_dict["rewards_sentence2"]

        # Stack logits and apply softmax
        logits = torch.stack([logits1, logits2], dim=1)  # Stack along dimension 1
        logits = logits.softmax(dim=1) # Apply softmax along dimension 1

        labels = torch.zeros(logits.shape[0])
        return loss, logits, labels
    
    def compute_loss(
        self, 
        model: nn.Module,
        inputs: List[Tuple[str, str, int]],
        return_outputs=False,
    ):
        sentence1, sentence2, polarity = inputs
        polarity = torch.tensor(polarity).detach().to('cuda')
        sentence1_output = model(sentence1)
        sentence2_output = model(sentence2)
        embeddings1 = sentence1_output["sentence_embedding"]
        embeddings2 = sentence2_output["sentence_embedding"]
        logits1 = sentence1_output["logits"][:, 0]
        logits2 = sentence2_output["logits"][:, 0]
        flip_reward_mask = torch.where(polarity == 0, torch.tensor(1), torch.tensor(-1))
        reward_weight_mask = torch.where(polarity == 0, self.reward_weight, 1 - self.reward_weight)
        contrastive_weight_mask = torch.where(polarity == 0, 1 - self.reward_weight, self.reward_weight)

        # Compute contrastive loss for sentences 
        distances = 1 - F.cosine_similarity(embeddings1, embeddings2)
        cl_loss = 0.5 * (
            polarity.float() * distances.pow(2) + (1 - polarity).float() * F.relu(self.margin - distances).pow(2)
        )
        cl_weighted = cl_loss * contrastive_weight_mask 

        # Compute reward loss for logits 
        logits2 = logits2 * flip_reward_mask 
        rm_loss = -nn.functional.logsigmoid(logits1 - logits2)
        rm_weighted = rm_loss * reward_weight_mask 
        loss = rm_weighted + cl_weighted

        loss = loss.mean() 
        if return_outputs:
            return loss, {
                "rewards_sentence1": logits1,
                "rewards_sentence2": logits2,
            }
        return loss