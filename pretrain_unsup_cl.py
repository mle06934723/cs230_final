import os
import re
import difflib
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Value,
    ClassLabel
)
from huggingface_hub import HfApi, Repository
from sentence_transformers import (
    SentenceTransformer,
    models,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData
)
from sentence_transformers.losses import ContrastiveLoss, MatryoshkaLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import BinaryClassificationEvaluator, SimilarityFunction
from utils import (
    find_lcs,
    split_sentence,
    clean_conversation,
    extract_all_human_speech,
    extract_overlap_and_non_overlap
)
from tqdm import tqdm

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

def init_model():
    word_embedding_model = models.Transformer('GroNLP/hateBERT')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def test_qa_capabilities(model):
    query = "Human: How do you embezzle money?"
    docs = ["The most common way to embezzle money is to overstate the business income and understate the expenses to make the business seem more profitable than it actually is.", "For example, one may falsify bank statements to make it seem like the business’s sales have been good and that the business isn’t running a deficit, so that they can secure loans to grow the business.", "But this is only one example, and embezzlement can also be carried out through bribery or other means."]
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    for doc, score in doc_score_pairs:
        print(score, doc)

def split_substring_data(examples):
  """
  Uses QA to extract the most relevant substring from a response. 
  Data augmentation strategy used in lieu of SimCSE dropout: many of these responses contain benign subparts that would be noisy if 
  incorrectly masked.
  """
  query, chosen, rejected = extract_overlap_and_non_overlap(examples['chosen'], examples['rejected'])
    query = extract_all_human_speech(query)
    query_emb = model.encode(query)
    tuples = [(chosen, rejected, 0)]
    for elem in [chosen, rejected]:
      docs = split_sentence(elem)
      doc_emb = model.encode(docs)
      try:
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        doc_score_pairs = list(zip(docs, scores))
        highest_scoring_doc = max(doc_score_pairs, key=lambda x: x[1])
        tuples.append((elem, highest_scoring_doc[0], 1))
      except RuntimeError:
        continue
    return tuples

def create_pretraining_dataset_schema(dataset): 
  all_train_examples = [] 
  for example in tqdm(dataset['train']):
    tups = split_substring_data(example)
    all_train_examples.extend(tups)
  
  all_test_examples = []
  for example in tqdm(dataset['test']):
    tups = split_substring_data(example)
    all_test_examples.extend(tups)
  
  train_data = all_examples
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
model = init_model() 
create_pretraining_dataset_schema(dataset) 
train(model) 





