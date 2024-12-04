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

    def map_similarity_to_confidence(self, similarity_scores, same_labels_mask):
        """
        This function takes similarity scores and maps them to confidence values.
        For same labels, the confidence is mapped closer to 1 as similarity increases.
        For different labels, the confidence is mapped closer to -1 as similarity decreases.
        """
        # Ensure both matrices are of the same shape
        assert same_labels_mask.shape == similarity_scores.shape, "Matrices must have the same shape."

        # Convert boolean mask to float (True becomes 1, False becomes 0)
        same_labels_mask_float = same_labels_mask.float()

        # Confidence when labels are the same: cosine similarity
        confidence_same = same_labels_mask_float * similarity_scores

        # Confidence when labels are different: 1 - cosine similarity
        confidence_diff = (1 - same_labels_mask_float) * (1 - similarity_scores)

        # The total confidence is the sum of both
        confidence = confidence_same + confidence_diff

        # Ensure confidence is between 0 and 1
        confidence = torch.clamp(confidence, 0, 1)

        # Set diagonal values to -1
        confidence = confidence.masked_fill(torch.eye(confidence.size(0), dtype=torch.bool, device=confidence.device), -1)

        # Convert to sparse COO format
        # Get indices of non-zero elements
        indices = confidence.nonzero().t()

        # Get values of non-zero elements
        values = confidence[indices[0], indices[1]]

        # Create the sparse tensor
        sparse_confidence = torch.sparse_coo_tensor(indices, values, confidence.size())

        return sparse_confidence


    def select_pairs(self, selected_examples, similar_graph_all, noisy_labels, device='cuda'):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            selected_examples = selected_examples.to(device).bool()  # Convert to boolean mask
            similarity_scores = similar_graph_all.to(device)
            shape = similarity_scores.shape
            print("PRINTING SIMILARITY SCORES WITHIN PAIR SEL")
            print(similarity_scores.max(), similarity_scores.min())
            noisy_labels = noisy_labels.to(device)

            total_num = len(noisy_labels)

            # Create masks for same and different labels
            self.same_labels_mask = torch.eq(noisy_labels.unsqueeze(0), noisy_labels.unsqueeze(1))  # True if labels match
            sparse_confidence = self.map_similarity_to_confidence(similarity_scores, self.same_labels_mask)
            sparse_confidence = sparse_confidence.coalesce()
            print("DONE MAPPING SIMILARITY TO CONFIDENCE")
            print(sparse_confidence.values().max(), sparse_confidence.values().min())
            print(sparse_confidence._nnz())

            # CREATE FILTER MASKS
            selected_mask = selected_examples.unsqueeze(0) & selected_examples.unsqueeze(1)
            non_diag_mask = ~torch.eye(shape[0], dtype=torch.bool, device=device)

            # FILTER 1: NON-DIAGONAL VALUES
            mask = non_diag_mask[sparse_confidence.indices()[0], sparse_confidence.indices()[1]]
            filtered_indices = sparse_confidence.indices()[:, mask]  # Filter indices where the mask is true
            filtered_values = sparse_confidence.values() [mask]  # Filter values that meet the mask condition
            sparse_confidence_matrix = torch.sparse_coo_tensor(filtered_indices, filtered_values, shape, dtype=torch.float16, device=device)
            sparse_confidence_matrix = sparse_confidence_matrix.coalesce()
            print(sparse_confidence_matrix.values().max(), sparse_confidence_matrix.values().min())
            print(sparse_confidence_matrix._nnz())
            """
            # FILTER 2: ZERO OUT ALL VALUES WHERE BOTH INDICES NOT IN SELECTED EXAMPLES
            mask = selected_mask[sparse_confidence_matrix.indices()[0], sparse_confidence_matrix.indices()[1]]
            filtered_indices = sparse_confidence_matrix.indices()[:, mask]
            filtered_values = sparse_confidence_matrix.values()[mask]
            sparse_confidence_matrix = torch.sparse_coo_tensor(filtered_indices, filtered_values, similarity_scores.shape)
            sparse_confidence_matrix = sparse_confidence_matrix.coalesce()
            print(sparse_confidence.values().max(), sparse_confidence.values().min())
            """

        del similarity_scores
        return sparse_confidence_matrix

    def pair_selection(self, train_features, trainloader):
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=1, shuffle=False, num_workers=8)
        init_noisy_labels = torch.LongTensor(temploader.dataset.targets)
        train_new_labels, similarity_graph_all = self.weighted_knn(temploader, train_features, init_noisy_labels, True)
        print(similarity_graph_all.max(), similarity_graph_all.min())
        agreement_measure, discrepancy_measure = self.weighted_knn(temploader, train_features, train_new_labels, False)
        selected_examples = self.select_examples(temploader, discrepancy_measure, agreement_measure)
        selected_pairs = self.select_pairs(selected_examples, similarity_graph_all, init_noisy_labels)
        return selected_examples, selected_pairs

    def order_pairs_curriculum(self, pairs):
        pairs = pairs.coalesce()
        # Extract the indices and values from the sparse tensor
        indices = pairs.indices()  # Shape: [2, num_nonzero_elements]
        values = pairs.values()    # Shape: [num_nonzero_elements]
        print(values.shape)
        polarity = self.same_labels_mask.float()

        # Ensure i <= j for all pairs (remove mirrored duplicates)
        i_indices = indices[0]
        j_indices = indices[1]

        i_indices, j_indices = torch.min(i_indices, j_indices), torch.max(i_indices, j_indices)

        pair_mask = i_indices <= j_indices

        # Apply the mask to filter out invalid pairs
        filtered_i = i_indices[pair_mask]
        filtered_j = j_indices[pair_mask]
        filtered_values = values[pair_mask]

        k = min(5000000, len(values))

        # Use topk to get the top k pairs based on the absolute values
        topk_values, topk_indices = torch.topk(filtered_values, k, largest=True, sorted=True)

        # Retrieve the corresponding indices for the top k values
        topk_i = filtered_i[topk_indices]
        topk_j = filtered_j[topk_indices]
        topk_polarity = polarity[topk_i, topk_j]

        unique_result = [(topk_i[i].item(), topk_j[i].item(), topk_polarity[i].item()) for i in range(k)]

        if unique_result:
          num_zero_polarity = sum(1 for _, _, p in unique_result if p == 0)
          num_one_polarity = sum(1 for _, _, p in unique_result if p == 1)
          print(f"Number of polarity elements equal to 0: {num_zero_polarity}")
          print(f"Number of polarity elements equal to 1: {num_one_polarity}")
          # First and last elements in unique_sorted_indices
          first_indices = unique_result[0]
          last_indices = unique_result[-1]

          # Retrieve the values from the `pairs` tensor at these indices
          first_value = pairs[first_indices[0], first_indices[1]].item()  # Using the row and column indices from the sorted pair
          last_value = pairs[last_indices[0], last_indices[1]].item()

          # Print the absolute values of the first and last elements
          print("First element: ", first_indices)
          print("First element confidence score:", first_value)
          print(f"First element {self.targets[first_indices[0]]} sentence1: {self.texts[first_indices[0]]}")
          print(f"First element {self.targets[first_indices[1]]} sentence1: {self.texts[first_indices[1]]}")
          print("Last element: ", last_indices)
          print(f"Last element {self.targets[last_indices[0]]} sentence2: {self.texts[last_indices[0]]}")
          print(f"Last element {self.targets[last_indices[1]]} sentence2: {self.texts[last_indices[1]]}")
          print(f"Last element confidence score:", last_value)
          print(f"Length of unique result: {len(unique_result)}")
        return unique_result

