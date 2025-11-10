import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from icecream import ic

class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size 
        self._mask_value = -1e9

        # Layer
        self.q_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )

        self.k_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )

        self.v_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )
    
    def forward(self, Q_input, K_input, V_input):
        """
            :params Q_input:    BS, number_of_element_q , input_size
            :params K_input:    BS, number_of_element_k , input_size
            :params V_input:    BS, number_of_element_v , input_size
        """
        Q = self.q_linear(Q_input)
        K = self.k_linear(K_input)
        V = self.v_linear(V_input)
        QK = QK = torch.bmm(
            Q, torch.transpose(K, 2, 1), 
        ) # BS, number_of_element_q, number_of_element_k
        A = QK / math.sqrt(self.input_size)
        
        return A, V
    

class DeFumAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self._mask_value = -1e9

        self.self_attention = SelfAttention(input_size)

    
    def forward(
        self,
        visual_entity,
        relative_depth_map,
        attention_mask
    ):
        A, V = self.self_attention(
            Q_input=visual_entity, 
            K_input=visual_entity, 
            V_input=visual_entity
        )
        scores = torch.bmm(F.softmax(A + relative_depth_map, dim=-1) + V)
        scores = scores.masked_fill(attention_mask == 0, self._mask_value)
        return scores





    



