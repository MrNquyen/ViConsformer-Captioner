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
        self._mask_value = -1e9

        #-- LAYER
        self.svoce = SVOCE()

        self.fasttext_dim = 300
        self.q_linear = nn.Linear(
            in_features=self.fasttext_dim,
            out_features=self.fasttext_dim
        )

        self.k_linear = nn.Linear(
            in_features=self.fasttext_dim,
            out_features=self.fasttext_dim
        )
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)

    
    def fasttext_embedding(self, words: list):
        """
            :params words:  List of word needed to embedded
        """
        fasttext_embedding = [
            fasttext_embedding_module(
                model=self.fasttext_model,
                word=word
            ) 
            for word in words
        ]
        return torch.tensor(fasttext_embedding).to(self.device)



    #-- Forward Pass
    def forward(self, batch):
        list_ocr_tokens = batch["list_ocr_tokens"]
        ocr_mask = batch["ocr_mask"]

        #-- Make Features
        visual_object_concept_feat, concepts_fasttext_feat = self.svoce(batch)
        ocr_tokens_fasttext_feat = torch.tensor([self.fasttext_embedding(tokens) for tokens in list_ocr_tokens]).to(self.device)
        
        #-- Scores Attention
        Q = self.q_linear(concepts_fasttext_feat)
        K = self.k_linear(ocr_tokens_fasttext_feat)
        QK = torch.bmm(
            Q, torch.transpose(K, 2, 1)
        )
        A = QK / math.sqrt(self.fasttext_dim)
        
        semantic_scores = torch.bmm(F.softmax(A, dim=-1), concepts_fasttext_feat)
        semantic_scores = semantic_scores.masked_fill(ocr_mask == 0, self._mask_value)

        #-- Semantic Embedding for OCR Tokens
        semantic_ocr_tokens_feat = self.LayerNorm(ocr_tokens_fasttext_feat + semantic_scores)
        return semantic_ocr_tokens_feat, visual_object_concept_feat


