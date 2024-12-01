import torch
import tqdm
import torch.nn.functional as F 
from torch.utils.data import Dataset
import numpy as np 
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

class CurriculumContrastiveRewardTrainDataset(Dataset): 
    def __init__(self, binary_dataset: BinaryRewardModelDataset, train_features, init_dataloader):
        self.binary_dataset = binary_dataset 
        self.num_classes = 2
        self.low_dim = 128
        self.k_val = 10
        self.alpha = 0.5
        self.beta = 0.25
        self.sup_t = 0.1
        self.targets = self.binary_dataset.targets 
        self.texts = self.binary_dataset.texts
        # Run pair selection algorithm to extract highly confident pairs represented as a 2D matrix
        self.selected_examples, pairs = self.pair_selection(train_features, init_dataloader)
        # Order the pairs by highest confidence 
        self.tuples = self.order_pairs_curriculum(pairs)
        # Within tuples, order chosen response (label = 1) always first if applicable
        self.ordered_tuples = self.order_chosen_first(self.tuples, self.binary_dataset)
        # Extract sentences from ordered tuples. This is what __getitem__ will yield from. 
        self.sentence_tuples = self.extract_sentences(self.ordered_tuples, self.binary_dataset)
        # Debug top 50 close (chosen, chosen or rejected, rejected) and far pairings (chosen, rejected) to sanity check 
        self.top_close, self.top_far = self.debug_top_elements(self.ordered_tuples)

    def __len__(self):
        return len(self.sentence_tuples)

    def __getitem__(self, index):
        """
        Returns the tuple at the given index in the reordered list.
        """
        return self.sentence_tuples[index]

    def order_chosen_first(self, tuples, dataset):
        # Extract sentence indices from tuples
        sentence1_indices = torch.tensor([t[0] for t in tuples], dtype=torch.long)
        sentence2_indices = torch.tensor([t[1] for t in tuples], dtype=torch.long)
        
        # Extract labels using dataset (based on sentence indices)
        sentence1_labels = [self.targets[i] for i in sentence1_indices]
        sentence2_labels = [self.targets[i] for i in sentence2_indices]
        
        # Create masks for sentences where label is 1 (chosen example)
        sentence1_is_label_1 = torch.tensor(sentence1_labels) == 1
        sentence2_is_label_1 = torch.tensor(sentence2_labels) == 1

        # Create a mask for cases where we need to swap sentence order
        swap_mask = sentence2_is_label_1 & ~sentence1_is_label_1  # sentence2 has label 1, sentence1 does not
        
        # Reorder indices based on the swap_mask
        sentence1_indices[swap_mask], sentence2_indices[swap_mask] = sentence2_indices[swap_mask], sentence1_indices[swap_mask]
        
        # Rebuild tuples with reordered indices
        new_tuples = list(zip(sentence1_indices.tolist(), sentence2_indices.tolist(), [t[2] for t in tuples]))
        
        return new_tuples

    def extract_sentences(self, tuples, dataset):
        # Extract sentence indices from tuples
        sentence1_indices = torch.tensor([t[0] for t in tuples], dtype=torch.long)
        sentence2_indices = torch.tensor([t[1] for t in tuples], dtype=torch.long)
        
        # Retrieve sentences
        sentence1_vals = [self.texts[i] for i in sentence1_indices]
        sentence2_vals = [self.texts[i] for i in sentence2_indices]
        
        # Return the sentences as a tuple (sentence1, sentence2, polarity)
        sent_tuples = list(zip(sentence1_vals, sentence2_vals, [t[2] for t in tuples]))
        
        return sent_tuples

    def weighted_knn(self, temploader, features, noisy_labels, return_labels, bsz = 1):
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
            retrieval_one_hot_train = torch.zeros(self.k_val, self.num_classes)

            for batch_idx, (data, labels, index) in tqdm(enumerate(temploader), total=len(temploader)):
                features_transpose = features.t()[index]
                dist = torch.mm(features_transpose, features)
                if return_labels: # if in first loop against original noisy labels, compute similarity_graph_all
                    similarity_graph_all[index] = dist.detach()
                # access diagonals of the matrix, or self
                dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  ## Self-contrast set to -1

                yd, yi = dist.topk(self.k_val, dim=1, largest=True, sorted=True)  ## Top-K similar scores and corresponding indexes
                candidates = new_labels.view(1, -1).expand(bsz, -1)  ## Replicate the labels per row to select
                retrieval = torch.gather(candidates, 1, yi)  ## Get top-K neighbour labels

                retrieval_one_hot_train.resize_(bsz * self.k_val, self.num_classes).zero_()
                retrieval_one_hot_train.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = torch.exp(yd.clone().div_(self.sup_t))  ## Apply temperature to scores
                yd_transform[...] = 1.0  ## To avoid using similarities
                probs_corrected = torch.sum(
                    torch.mul(retrieval_one_hot_train.view(bsz, -1, self.num_classes),
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


    def select_examples(self, temploader, final_discrepancy_measure, agreement_measure):
        num_clean_per_class = torch.zeros(self.num_classes)
        targets = torch.tensor(temploader.dataset.targets).squeeze()  # Shape: [N] (1D tensor of class labels)

        # Step 1: Count clean examples for each class
        for i in range(self.num_classes):
            idx_class = targets == i
            num_clean_per_class[i] = torch.sum(agreement_measure[idx_class])

        # Step 2: Calculate median number of clean examples per class
        num_samples2select_class = torch.median(num_clean_per_class.float()).item()

        # Reset the agreement measure to zero
        agreement_measure = torch.zeros((len(targets),))

        # Step 3: Select examples for each class
        for i in range(self.num_classes):
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
      
    def debug_top_elements(self, tuples):
      # Separate tuples based on the value of 'val'
      top_ones = [t for t in tuples if t[2] == 1]  # Filter where val == 1, indicating close match
      top_zeros = [t for t in tuples if t[2] == 0]  # Filter where val == 0, indicating far match

      # Get the top 50 elements from each list (if available)
      top_ones = top_ones[:50]
      top_zeros = top_zeros[:50]

      return top_ones, top_zeros

    def map_similarity_to_confidence(self, similarity_scores, same_labels_mask, different_labels_mask):
        """
        This function takes similarity scores and maps them to confidence values.
        For same labels, the confidence is mapped closer to 1 as similarity increases.
        For different labels, the confidence is mapped closer to -1 as similarity decreases.
        """
        non_self_mask = ~torch.eye(similarity_scores.size(0), dtype=torch.bool, device=similarity_scores.device)

        # For same labels: high similarity corresponds to high positive confidence (close to 1)
        confidence_same = similarity_scores * 2 - 1  

        # For different labels: low similarity corresponds to high negative confidence (close to -1)
        confidence_diff = -(similarity_scores * 2 - 1)  

        # Initialize a confidence matrix with zeros
        confidence = torch.zeros_like(similarity_scores)

        # Apply the confidence mapping for same labels
        confidence[same_labels_mask & non_self_mask] = confidence_same[same_labels_mask & non_self_mask]

        # Apply the confidence mapping for different labels
        confidence[different_labels_mask & non_self_mask] = confidence_diff[different_labels_mask & non_self_mask]

        return confidence

    def select_pairs(self, selected_examples, similar_graph_all, noisy_labels, device='cuda'):
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
            confidence_matrix = self.map_similarity_to_confidence(similarity_scores, same_labels_mask, different_labels_mask)
            # Compute the threshold value based on the quantile of similarities in the selected mask
            selected_similarity_scores = similarity_scores[selected_mask]
            # Split the tensor into smaller chunks
            chunk_size = 10000  # Adjust this size based on your memory capacity
            selected_threshold = []
            for i in range(0, len(selected_similarity_scores), chunk_size):
                chunk = selected_similarity_scores[i:i + chunk_size]
                selected_threshold.append(torch.quantile(chunk, self.beta))
            selected_threshold = torch.mean(torch.tensor(selected_threshold))  # Average the quantiles

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

    def pair_selection(self, train_features, trainloader):
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=1, shuffle=False, num_workers=8)
        init_noisy_labels = torch.LongTensor(temploader.dataset.targets)
        train_new_labels, similarity_graph_all = self.weighted_knn(temploader, train_features, init_noisy_labels, True)
        agreement_measure, discrepancy_measure = self.weighted_knn(temploader, train_features, train_new_labels, False)
        selected_examples = self.select_examples(temploader, discrepancy_measure, agreement_measure)
        selected_pairs = self.select_pairs(selected_examples, similarity_graph_all, init_noisy_labels)
        return selected_examples, selected_pairs

    def order_pairs_curriculum(self, pairs):
        # Get absolute values and mask the diagonal to ignore i == j pairs
        abs_pairs = pairs.abs()
        mask = torch.eye(pairs.shape[0], device=pairs.device, dtype=torch.bool)  # Exclude diagonal elements
        abs_pairs = abs_pairs.masked_fill(mask, -float('inf'))  # Set diagonal elements to a very low value

        # Get the sorted indices based on the absolute values in descending order
        sorted_indices = torch.argsort(abs_pairs.view(-1), descending=True)

        # Compute the indices (i, j) from sorted linear indices
        i_indices = sorted_indices // pairs.shape[1]
        j_indices = sorted_indices % pairs.shape[1]

        # Filter out i == j (self-pairs)
        valid_mask = i_indices != j_indices

        i_indices = i_indices[valid_mask]
        j_indices = j_indices[valid_mask]

        # Get values from the original tensor
        values = pairs[i_indices, j_indices]

        # Filter out zero-value pairs
        non_zero_mask = values != 0.0
        i_indices = i_indices[non_zero_mask]
        j_indices = j_indices[non_zero_mask]
        polarities = (values[non_zero_mask] > 0).long()

        # Efficiently create a unique mask for mirrored pairs (i, j) and (j, i)
        pairs_tuple = torch.stack([i_indices, j_indices], dim=1)
        sorted_pairs = torch.sort(pairs_tuple, dim=1)[0]  # Sort pairs lexicographically to avoid mirrored pairs

        # Create a mask for unique pairs (by using tuple sorting)
        _, unique_indices = torch.unique(sorted_pairs, dim=0, return_inverse=True)

        # Gather the unique pairs using the indices
        unique_i_indices = i_indices[unique_indices]
        unique_j_indices = j_indices[unique_indices]
        unique_polarities = polarities[unique_indices]

        # Combine results into a list of tuples
        mapped_polarities = list(zip(unique_i_indices.cpu().numpy(), unique_j_indices.cpu().numpy(), unique_polarities.cpu().numpy()))

        return list(dict.fromkeys(mapped_polarities))


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

