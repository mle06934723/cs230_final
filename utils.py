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