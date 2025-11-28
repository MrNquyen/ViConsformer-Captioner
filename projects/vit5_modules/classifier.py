import torch
from torch import nn

from utils.registry import registry

class Classifier(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
    
    def forward(self, dec_last_hidden_state):
        return self.classifier(dec_last_hidden_state)

    def get_vocab_size(self):
        return self.classifier.weight.size(0)