import os
import torch
import numpy as np
from torch import nn

from utils.registry import registry
from utils.utils import load_json, load_npy, load_vocab
from utils.module_utils import fasttext_embedding_module
from icecream import ic

# Salient Visual Object Concepts Extractor
class SVOCE(nn.Module):
    def __init__(self, fasttext_model):
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.svoce_config = self.config["svoce"]
        self.dataset_attributes_config = registry.get_config("dataset_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        self.fasttext_model = fasttext_model
        self.hidden_size = self.config["hidden_size"]
        self.common_dim = self.config["feature_dim"]
        self.top_k = self.svoce_config["top_k"]

        #--layer
        fasttext_dim = 300
        self.linear_concept = nn.Linear(
            in_features=fasttext_dim,
            out_features=self.hidden_size
        )
        self.LayerNorm_voc = nn.LayerNorm(normalized_shape=self.hidden_size)

        self.linear_score = nn.Linear(
            in_features=1,
            out_features=self.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)

        #-- Load modules
        self.load()
        
    #-- BUILD
    def load(self):
        self.load_object_concepts()
        self.load_object_concepts_feat()


    def load_object_concepts(self):
        self.object_concepts_classes_en = load_vocab(self.dataset_attributes_config["object_concepts"]["object_concepts_en"])
        self.object_concepts_classes_vi = load_vocab(self.dataset_attributes_config["object_concepts"]["object_concepts_vi"])
        assert len(self.object_concepts_classes_en) == len(self.object_concepts_classes_vi)
        self.en2vi = {
            en_word: vi_word
            for en_word, vi_word in zip(self.object_concepts_classes_en, self.object_concepts_classes_vi)
        }
        self.id2en = {
            id: en_word
            for id, en_word in enumerate(self.object_concepts_classes_en)
        }


    def load_object_concepts_feat(self):
        self.clip_object_concepts_feat_dir = self.svoce_config["object_concepts_features"]
        clip_object_concepts_feat_path_template = os.path.join(self.clip_object_concepts_feat_dir, "{concept_name}.npy")

        self.word2feat = {}
        for en_word in self.object_concepts_classes_en:
            clip_feat_path = clip_object_concepts_feat_path_template.format(concept_name=en_word)
            concept_clip_feat = load_npy(clip_feat_path)
            self.word2feat[en_word] = torch.tensor(concept_clip_feat).unsqueeze(0)
        all_clip_feat = list(self.word2feat.values())
        self.clip_concepts_feat = torch.concat(all_clip_feat, dim=0).to(self.device) # 600, clip_hidden_size

    
    def calculate_scores(self, list_clip_image_feat):
        BATCH_SIZE = list_clip_image_feat.size(0)
        ic(self.clip_concepts_feat.shape)
        expand_clip_concepts_feat = self.clip_concepts_feat.expand(BATCH_SIZE, -1, -1)
        ic(expand_clip_concepts_feat.shape)
        scores = torch.bmm(
            expand_clip_concepts_feat, torch.transpose(list_clip_image_feat, 2, 1)
        ).softmax(dim=-1)
        return scores.squeeze(-1)
    

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
        list_clip_image_feat = batch["list_clip_image_feat"] # BS, 1, clip_hidden_size
        ic(list_clip_image_feat.shape)
        similarity_scores = self.calculate_scores(list_clip_image_feat) # BS, num_concepts, 1

        #-- Sort
        top_k_concepts_id = torch.argsort(similarity_scores, dim=1, descending=True).squeeze(-1)[:, :self.top_k]
        top_k_concepts_vi = [[self.en2vi[self.id2en[id.item()]] for id in per_image_top_k] for per_image_top_k in top_k_concepts_id]
        ic(similarity_scores.shape)
        ic(top_k_concepts_id.shape)
        top_k_concepts_scores = torch.gather(similarity_scores, 1, top_k_concepts_id).unsqueeze(-1)

        #-- Fasttext Embedding
        concepts_fasttext_feat = torch.stack([self.fasttext_embedding(concepts) for concepts in top_k_concepts_vi]).to(self.device)

        #-- Visual Object Concept Embedding
        ic(top_k_concepts_scores.shape)
        visual_object_concept_feat = self.LayerNorm(
            self.linear_concept(concepts_fasttext_feat)
        ) + self.LayerNorm(
            self.linear_score(top_k_concepts_scores)
        )
        return visual_object_concept_feat, concepts_fasttext_feat
    
    