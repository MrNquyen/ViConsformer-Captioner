import os
import torch
import numpy as np
from torch import nn

from utils.registry import registry
from utils.utils import load_json, load_npy, load_vocab
from utils.module_utils import fasttext_embedding_module
from projects.modules.svoce import SVOCE
from icecream import ic

# Salient Visual Object Concepts Extractor
class SgAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.sgam_config = self.config["sgam"]
        self.dataset_attributes_config = self.config["dataset_attributes"]
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        self.hidden_size = self.config["hidden_size"]
        self.common_dim = self.config["feature_dim"]

        #-- LAYER
        self.svoce = SVOCE()
        self.q_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )

        self.k_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )


    #-- Forward Pass
    def forward(self, batch):
        list_ocr_tokens = batch["list_ocr_tokens"]

        #-- Make Features
        visual_object_concept_feat = torch.tensor([self.fasttext_embedding(concepts) for concepts in self.object_concepts_classes_vi]).to(self.device)
        ocr_tokens_fasttext_feat = torch.tensor([self.fasttext_embedding(tokens) for tokens in list_ocr_tokens]).to(self.device)
        
        #-- Attention
