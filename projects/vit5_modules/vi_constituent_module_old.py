import torch
import math
from torch import nn
from torch.nn import functional as F
from vit5_modules.base import BaseEmbedding

from utils.registry import registry


class ViConstituentModuleOld(nn.Module):
    def __init__(self):
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.num_ocr = self.config["ocr"]["num_ocr"]
        self.hidden_state = self.config["hidden_state"]

        self.linear_next_token = nn.Linear(self.hidden_state, self.hidden_state)

    def cal_semantic_relation(
        self,
        cur_ocr_features,
        next_ocr_features
    ):
        semantic_relations = []
        for i in range(self.num_ocr - 1):
            rel = torch.bmm(
                cur_ocr_features[:, i, :].unsqueeze(1),
                torch.transpose(linear_next_token_out[:, i, :].unsqueeze(1), 2, 1)
            )
            semantic_relations.append(rel)
        semantic_relations = torch.concat(semantic_relations, dim=1)
        return semantic_relations # BS, num_ocr - 1, 1


    def forward(self, batch, vit5_ocr_tokens_embed, vit5_ocr_tokens_embed_mask):
        batch_size = vit5_ocr_tokens_embed.size(0)

        cur_ocr_features = ocr_features[:, :-1, :]
        next_ocr_features = ocr_features[:, 1:, :]

        #-- Finding semactic relations
        linear_next_token_out = self.linear_next_token(next_ocr_features)
        semantic_relations = self.cal_semantic_relation(
            cur_ocr_features= cur_ocr_features,
            next_ocr_features= linear_next_token_out
        )

        #-- Fist word has no prev word, last word has no next word
        no_semantic_relations_feat = torch.full((batch_size, 1, 1), -10000)
        semantic_relations = torch.concat(
            [
                no_semantic_relations_feat, #~ Fist word has no prev word
                semantic_relations, 
                no_semantic_relations_feat #~  Last word has no next word
            ], dim=1
        ) # BS, num_ocr + 1, 1
        semantic_relations = semantic_relations.squeeze(-1)

        #-- Calculate prob that prev and next word contribute to current word
        A = semantic_relations[:, :-1, :] #~ Prev word
        B = semantic_relations[:, 1:, :] #~ Next word
        pairs = torch.stack([A, B], dim=1)
        prob_semantic_relation_pairs = F.softmax(pairs, dim=1) # BS, num_ocr, 2 (prob for prev and next word base on current word)

        #                 a           b            c           d   
        #                k-1          k           k+1         k+2
        #       empty       r_k-1_k      r_k_k+1      r_k+1_k+2       empty
        #       
        #       [
        #           pr_k-1_k-2    pr_k-1_k     pr_k_k+1
        #            pr_k-1_k     pr_k_k+1    pr_k_1_k+2           
        #       ]

        C = torch.zeros((batch_size, num_ocr, 2)) #~ 2 is the prob of prev and next




class ViConstituentModuleOld(nn.Module):
    def __init__(self):
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.num_ocr = self.config["ocr"]["num_ocr"]
        self.hidden_state = self.config["hidden_state"]

        self.linear_next_token = nn.Linear(self.hidden_state, self.hidden_state)

    def forward(self, self, batch, vit5_ocr_tokens_embed, vit5_ocr_tokens_embed_mask):
        batch_size = vit5_ocr_tokens_embed.size(0)
        num_ocr = vit5_ocr_tokens_embed.size(1)

        a = torch.diag(torch.ones(num_ocr - 1), 1).long().to(self.device)
        b = torch.diag(torch.ones(num_ocr), 0).long().to(self.device)
        c = torch.diag(torch.ones(num_ocr - 1), -1).long().to(self.device)

        



    


