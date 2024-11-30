import torch 

class IntegrityClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes, hidden_size, low_dim):
        super(IntegrityClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dim_reducer = nn.Linear(hidden_size, low_dim)

    def forward(self, inputs: Dict[str, torch.Tensor], feat_classifier=False):
        out = self.encoder(**inputs)
        out_ = out.pooler_output
        if(feat_classifier):
            return out_
        outContrast = self.dim_reducer(out_)
        outPred = self.classifier(out_)
        return outPred, outContrast
