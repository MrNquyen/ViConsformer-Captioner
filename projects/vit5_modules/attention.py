import torch
import math
from torch import nn
from torch.nn import functional as F
from vit5_modules.base import BaseEmbedding

from utils.registry import registry

# ================ SpartialCirclePosition =============================
class SpartialCirclePosition(nn.Module):
    def __init__(self):
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        self.spatial_pos_config = self.config["spatial_circle_position"]
        self.num_grid_patches = self.spatial_pos_config["num_grid_patches"]
        self.num_distances = self.spatial_pos_config["num_distances"]
        self.num_heads = self.spatial_pos_config["num_heads"]
        self.hidden_size = self.config["hidden_size"]
        self._mask_value = -1e9

        self.distance_embedding = nn.Embedding(
            num_embeddings=self.num_distances,
            embedding_dim=self.num_heads
        )

        self.head_dim = self.hidden_size // self.num_heads
        self.q_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)


    def calculate_2D_distances(
        self,
        selected_x_centroids: torch.Tensor, 
        selected_y_centroids: torch.Tensor
    ):
        """
            Euclid distances
            Args:
                selected_x_centroids (torch.Tensor): (BS, M)
                selected_y_centroids (torch.Tensor): (BS, M)
        """
        dx = selected_x_centroids.unsqueeze(-1) - selected_x_centroids.unsqueeze(1) # (BS, M, M) 
        dy = selected_y_centroids.unsqueeze(-1) - selected_y_centroids.unsqueeze(1) # (BS, M, M) 
        distances = (torch.sqrt(dx**2 + dy**2) * 2).long()
        distances = torch.clamp(
            distances, 
            min=0,
            max=self.num_distances - 1,
        )
        return distances # (BS, M, M) 
            
            
    def patch_selection(
        self,
        image_sizes: torch.Tensor,
        boxes: torch.Tensor,
        num_grid_patches: int
    ):
        """
            Args:
                image_sizes: BS, 2 (w, h)
                boxes: BS, M, 4
                num_grid_patches (int): Num patch in single grid
        """
        M = boxes.size(1)
        batch_size = image_sizes.size(0)
        size_per_patch = image_sizes // num_grid_patches # BS, 2

        #-- Setting lower, upper bounds for width and height
        #-- Lowest height and width is 0
        patch_indexs = torch.arange(start=0, end=num_grid_patches, step=1) # Index of patch on grid
        patch_indexs = patch_indexs.unsqueeze(0).repeat(batch_size, 1)
        lower_patch_indexs = patch_indexs[:, :-1]
        upper_patch_indexs = patch_indexs[:, 1:]

        width_lower_bounds = lower_patch_indexs * size_per_patch[:, 0].unsqueeze(-1)
        width_upper_bounds = upper_patch_indexs * size_per_patch[:, 0].unsqueeze(-1)
        height_lower_bounds = lower_patch_indexs * size_per_patch[:, 1].unsqueeze(-1)
        height_upper_bounds = upper_patch_indexs * size_per_patch[:, 1].unsqueeze(-1)
        
        width_lower_bounds = width_lower_bounds.unsqueeze(1).expand(-1, M, -1) # BS, M, num_grid_patches
        width_upper_bounds = width_upper_bounds.unsqueeze(1).expand(-1, M, -1) # BS, M, num_grid_patches
        height_lower_bounds = height_lower_bounds.unsqueeze(1).expand(-1, M, -1) # BS, M, num_grid_patches
        height_upper_bounds = height_upper_bounds.unsqueeze(1).expand(-1, M, -1) # BS, M, num_grid_patches

        #-- Finding centroids for ocr boxes
        x_centroids = ((boxes[:, :, 0] + boxes[:, :, 2]) // 2).unsqueeze(-1) 
        y_centroids = ((boxes[:, :, 1] + boxes[:, :, 3]) // 2).unsqueeze(-1) 
        
        #-- Finding which patches coordinates, each ocr tokens belong to
        selected_x_centroids = torch.logical_and(torch.le(width_lower_bounds, x_centroids), torch.le(x_centroids, width_upper_bounds)) # (bs, n_ocr, 11)
        selected_y_centroids = torch.logical_and(torch.le(height_lower_bounds, y_centroids), torch.le(y_centroids, height_upper_bounds)) # (bs, n_ocr, 11)
        selected_x_centroids = selected_x_centroids.to(torch.float).argmax(dim=-1) # (BS, M)
        selected_y_centroids = selected_y_centroids.to(torch.float).argmax(dim=-1) # (BS, M)

        #-- selected_x_centroids, selected_y_centroids are the centroids of patch which ocr boxes belong to
        return selected_x_centroids, selected_y_centroids

    def forward(self, batch, features, list_boxes, features_mask):
        list_im_width = batch["list_im_width"]
        list_im_height = batch["list_im_height"]

        image_sizes = torch.stack([
            list_im_width, 
            list_im_height
        ], dim=-1)

        #-- Patch Selection
        selected_x_centroids, selected_y_centroids = self.patch_selection(
            image_sizes=image_sizes, 
            boxes=list_boxes, 
            num_grid_patches=self.num_grid_patches
        )

        #-- Calculate Distance
        spatial_distances = self.calculate_2D_distances(selected_x_centroids, selected_y_centroids) # BS, M, M
        spatial_distances_embed = self.distance_embedding(spatial_distances) # # BS, M, M, num_head

        #-- Self-attention
        M = list_boxes.size(1)
        batch_size = image_sizes.size(0)

        Q = self.q_linear(features).view(batch_size, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.q_linear(features).view(batch_size, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.q_linear(features).view(batch_size, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        QK = torch.bmm(
            Q, torch.transpose(K, 3, 2), 
        ) # BS, num_heads, M, M
        A = QK / math.sqrt(self.head_dim)
        spatial_A = A + spatial_distances_embed.permute(0, 3, 2, 1) # BS, num_heads, M, M
        spatial_A = torch.softmax(spatial_A, dim=-1) # BS, num_heads, M, M
        
        spatial_att = torch.matmul(spatial_A, V) # BS, num_heads, M, head_dim
        spatial_att = spatial_att.permute(0, 2, 1, 3).view(batch_size, M, self.num_heads * self.head_dim) # BS, M, hidden_size
        spatial_att = spatial_att.masked_fill(features_mask == 0, self._mask_value)
        return spatial_att


# ================ ScaledDotProductAttention =============================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.config = registry.get_config("model_attributes")

        self.num_heads = num_heads
        self.d_model = d_model
        self._mask_value = -1e9

        assert self.d_model % self.num_heads == 0
        self.head_dim = self.d_model // self.num_heads
        self.q_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)

    
    def forward(
        self, 
        queries,
        keys,
        values,
        attn_gate,
        attn_mask
    ):
        batch_size = queries.size(0)
        q_len = queries.size(1)
        k_len = keys.size(1)

        Q = self.q_linear(features).view(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.q_linear(features).view(batch_size, k_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.q_linear(features).view(batch_size, k_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        QK = torch.bmm(
            Q, torch.transpose(K, 3, 2), 
        ) # BS, num_heads, q_len, k_len
        A = QK / math.sqrt(self.head_dim)
        modified_A = A + attn_gate.permute(0, 3, 2, 1) # BS, num_heads, q_len, k_len
        modified_A = modified_A.masked_fill(attn_mask == 0, self._mask_value)
        modified_A = torch.softmax(modified_A, dim=-1) # BS, num_heads, q_len, k_len
        
        modified_att = torch.matmul(modified_A, V) # BS, num_heads, q_len, head_dim
        modified_att = modified_att.permute(0, 2, 1, 3).view(batch_size, q_len, self.num_heads * self.head_dim) # BS, q_len, hidden_size
        return modified_att

        
        




# ================ ViConstituentModule =============================
class ViConstituentModule(nn.Module):
    def __init__(self):
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.config = registry.get_config("model_attributes")
        self.constituent_config = self.config["constituent_module"]
        self.num_ocr = self.config["ocr"]["num_ocr"]
        self.hidden_state = self.config["hidden_state"]
        self.num_heads = self.constituent_config["num_heads"]

        self.head_dim = self.hidden_size // self.num_heads
        self._mask_value = -1e9
        self.eps = 1e-4
        self.q_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_linear = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)


    def forward(self, self, batch, ocr_features, ocr_features_mask):
        """
            Args:
                ocr_features (BS, M, hidden_size) 
                ocr_features_mask (BS, M)
        """
        batch_size = ocr_features.size(0)
        num_ocr = ocr_features.size(1)

        Q = self.q_linear(ocr_features).view(batch_size, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # BS, num_heads, M, hidden_states 
        K = self.q_linear(ocr_features).view(batch_size, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # BS, num_heads, M, hidden_states 
        V = self.q_linear(ocr_features).view(batch_size, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # BS, num_heads, M, hidden_states 

        #-- Constituent Module
        prev_word_mask = torch.diag(torch.ones(num_ocr - 1), 1).long().to(self.device)
        main_diag_mask = torch.diag(torch.ones(num_ocr), 0).long().to(self.device)
        next_word_mask = torch.diag(torch.ones(num_ocr - 1), -1).long().to(self.device)
        prev_next_word_mask = prev_word_index + next_word_index # Annotate which word is prev and next word

        neibor_attn_mask = torch.logical_and(
            prev_next_word_mask, # (BS, M, M) 
            ocr_features_mask.unsqueeze(1) # (BS, 1, M) 
        ) # (BS, M, M)

        neibor_attn = torch.bmm(
            Q, K.transpose(3, 2)
        ) / self.head_dim
        neibor_attn = neibor_attn.masked_fill(neibor_attn_mask == 0, self._mask_value)
        neibor_attn = torch.softmax(neibor_attn, dim=-1) # (BS, num_heads, M, M)
            #~ Nhân cho đối xứng của nó - Tuy nhiên sẽ tạo ra 2 lần nhân đối xứng => Cần chia 2 khi cummulative sum
        neibor_attn = torch.sqrt(neibor_attn * neibor_attn.transpose(3, 2)+ self.eps) # Avoid sqrt for 0
        neibor_attn = prior + (1. - prior) * neibor_attn # (BS, num_heads, M, M)

            #~ Cummulative sum
        tri_matrix = torch.triu(torch.ones(seq_len, seq_len), diagonal = 0).float().to(self.device)
        tri_matrix = tri_matrix.unsqueeze(0).unsqueeze(0)
        t = torch.log(neibor_attn + 1e-9).masked_fill(prev_word_mask == 0, 0).matmul(tri_matrix)
        g_attn = tri_matrix.matmul(t).exp().masked_fill((tri_matrix.int() - main_diag_mask) == 0, 0)
        g_attn = g_attn + g_attn.transpose(-2, -1) + neibor_attn.masked_fill(main_diag_mask == 0, 1e-4)            
        return g_attn, neibor_attn


        