import torch 
import torch.nn as nn
from sentence_transformers import (
    SentenceTransformer,
    models,
)
from transformers import AutoModel, AutoTokenizer

class IntegrityClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes, hidden_size, low_dim):
        super(IntegrityClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dim_reducer = nn.Linear(hidden_size, low_dim)

    def forward(self, inputs: Dict[str, torch.Tensor], feat_classifier=False):
        out = self.encoder(**inputs)
        out_ = out.last_hidden_state[:, 0, :]
        if(feat_classifier):
            return out_
        small_feat = self.dim_reducer(out_)
        pred = self.classifier(out_)
        return pred, small_feat

def init_sentence_pretrained_hateBERT():
    word_embedding_model = models.Transformer('GroNLP/hateBERT')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def init_sentence_unsup_cl_rlhf_hateBERT():
    return SentenceTransformer("mleshen22/hateBERT-cl-rlhf-5-epochs")

def init_sentence_pretrained_bert_base_uncased():
    return SentenceTransformer("sentence-transformers/stsb-bert-base")

def init_sentence_unsup_cl_rlhf_bert_base_uncased():
    return SentenceTransformer("mleshen22/bert-base-uncased-cl-rlhf-5-epochs")

def init_vanilla_unsup_cl_rlhf_hateBERT(low_dim=768, hidden_size=768, num_clases=2): 
    model = AutoModel.from_pretrained("mleshen22/hateBERT-cl-rlhf-5-epochs", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("mleshen22/hateBERT-cl-rlhf-5-epochs")
    classifier = IntegrityClassificationModel(model, num_classes, hidden_size, low_dim)
    return model, tokenizer, classifier 

def init_vanilla_hateBERT(low_dim=768, hidden_size=768, num_clases=2): 
    pretrained_hatebert_automodel = AutoModel.from_pretrained("GroNLP/hateBERT", output_hidden_states=True)
    pretrained_hatebert_autotokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    pretrained_hatebert_integrityclassifier = IntegrityClassificationModel(pretrained_hatebert_automodel, num_classes, hidden_size, low_dim)
    return pretrained_hatebert_automodel, pretrained_hatebert_autotokenizer, pretrained_hatebert_integrityclassifier

def init_vanilla_unsup_cl_rlhf_bert_base_uncased(low_dim=768, hidden_size=768, num_clases=2): 
    finetuned_bert_automodel = AutoModel.from_pretrained("mleshen22/bert-base-uncased-cl-rlhf-5-epochs", output_hidden_states=True)
    finetuned_bert_autotokenizer = AutoTokenizer.from_pretrained("mleshen22/bert-base-uncased-cl-rlhf-5-epochs")
    finetuned_bert_integrityclassifier = IntegrityClassificationModel(finetuned_bert_automodel, num_classes, hidden_size, low_dim)
    return finetuned_bert_automodel, finetuned_bert_autotokenizer, finetuned_bert_integrityclassifier

def init_vanilla_bert_base_uncased(low_dim=768, hidden_size=768, num_clases=2): 
    pretrained_bert_automodel = AutoModel.from_pretrained("sentence-transformers/stsb-bert-base", output_hidden_states=True)
    pretrained_bert_autotokenizer = AutoTokenizer.from_pretrained("sentence-transformers/stsb-bert-base")
    pretrained_bert_integrityclassifier = IntegrityClassificationModel(pretrained_bert_automodel, num_classes, hidden_size, low_dim)
    return pretrained_bert_automodel, pretrained_bert_autotokenizer, pretrained_bert_integrityclassifier