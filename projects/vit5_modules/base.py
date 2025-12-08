import torch
from torch import nn

from utils.registry import registry

class BaseEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        self.hidden_size = self.config["hidden_size"]
        self.common_dim = self.config["feature_dim"]
        
    
    def forward(self, batch):
        NotImplemented
