import re
import difflib 
import util 

def find_lcs(chosen, rejected):
    sequence_matcher = difflib.SequenceMatcher(None, chosen, rejected)
    substring = sequence_matcher.find_longest_match(0, len(chosen), 0, len(rejected))
    return chosen[substring.a:substring.a + substring.size]

def extract_overlap_and_non_overlap(chosen, rejected):
    query = find_lcs(chosen, rejected)
    return query, chosen.replace(query, ""), rejected.replace(query, "")

def split_sentence(text):
    sentence_endings = r'[.!?;:…,]+'
    sentences = [s.strip() for s in re.split(sentence_endings, text) if s.strip()]
    return sentences

def clean_conversation(text):
    cleaned_text = re.sub(r'\b(Human|Assistant):\s*', '', text)
    return cleaned_text.strip()

def extract_all_human_speech(text):
    matches = re.findall(r'Human:\s*(.*)', text.strip())
    concatenated_text = ' '.join(matches)
    return concatenated_text

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

def split_substring_data(model, examples):
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

def compute_reward_loss(
    model: Union[PreTrainedModel, nn.Module],
    inputs: Dict[str, Union[torch.Tensor, Any]],
    return_outputs=False,
    num_items_in_batch=None,
) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """
    From TRL reward_trainer 
    """
    rewards_chosen = model(
        input_ids=inputs["input_ids_chosen"],
        attention_mask=inputs["attention_mask_chosen"],
        return_dict=True,
    )["logits"]
    rewards_rejected = model(
        input_ids=inputs["input_ids_rejected"],
        attention_mask=inputs["attention_mask_rejected"],
        return_dict=True,
    )["logits"]
    loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

    if self.args.center_rewards_coefficient is not None:
        loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

    if return_outputs:
        return loss, {
            "rewards_chosen": rewards_chosen,
            "rewards_rejected": rewards_rejected,
        }
    return loss

def compute_online_contrastive_loss(
    model,
    sentence_features: Iterable[Dict[str, torch.Tensor]], 
    labels: torch.Tensor, size_average=False,
    margin = 0.5,
) -> torch.Tensor: 
    """
    From SentenceTransformer online contrastive loss 
    """
    embeddings = [model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

    distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
    negs = distance_matrix[labels == 0]
    poss = distance_matrix[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss
