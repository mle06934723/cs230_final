import re
import difflib 

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
