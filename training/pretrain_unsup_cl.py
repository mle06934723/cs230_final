import os
import re
import difflib
import random 
import torch 
import np 
from datasets import (
    load_dataset,
)
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData
)
from sentence_transformers.losses import ContrastiveLoss, MatryoshkaLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import BinaryClassificationEvaluator, SimilarityFunction
from tqdm import tqdm
from data import create_pretraining_dataset_schema
from models import init_sentence_pretrained_hateBERT

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

run_name = "hateBERT-cl-rlhf-5-epochs"
dataset_name = "unsup_cl_anthropic_rlhf_hateBERT"

def train(model): 
  train_dataset = load_dataset(f"mleshen22/{dataset_name}", split="train")
  dev_dataset = load_dataset(f"mleshen22/{dataset_name}", split="dev")
  test_dataset = load_dataset(f"mleshen22/{dataset_name}", split="test")
  
  loss = ContrastiveLoss(model)
  loss = MatryoshkaLoss(model, loss, [768, 512, 256, 128, 64])
  
  args = SentenceTransformerTrainingArguments(
      # Required parameter:
      output_dir=f"models/{run_name}",
      # Optional training parameters:
      num_train_epochs=5,
      seed=42,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      learning_rate=2e-5,
      warmup_ratio=0.1,
      fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
      bf16=False,  # Set to True if you have a GPU that supports BF16
      batch_sampler=BatchSamplers.BATCH_SAMPLER,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
      # Optional tracking/debugging parameters:
      eval_strategy="epoch",
      save_strategy="epoch",
      save_total_limit=10,
      logging_steps=100,
      load_best_model_at_end=True,
      run_name=run_name,  # Will be used in W&B if `wandb` is installed
  )
  dev_evaluator = BinaryClassificationEvaluator(
      sentences1=dev_dataset["sentence1"],
      sentences2=dev_dataset["sentence2"],
      labels=dev_dataset["score"],
      name="all-rlhf-dev",
  )
  dev_evaluator(model)

  trainer = SentenceTransformerTrainer(
      model=model,
      args=args,
      train_dataset=train_dataset,
      eval_dataset=dev_dataset,
      loss=loss,
      evaluator=dev_evaluator,
  )
  trainer.train()

  test_evaluator = BinaryClassificationEvaluator(
      sentences1=test_dataset["sentence1"],
      sentences2=test_dataset["sentence2"],
      labels=test_dataset["score"],
      name="all-rlhf-test",
  )
  test_evaluator(model)

  model.save_pretrained(f"models/{run_name}/final")
  model.push_to_hub(f"{run_name}", exist_ok=True)

dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
model = init_sentence_pretrained_hateBERT() 
create_pretraining_dataset_schema(dataset, run_name) 
train(model) 





