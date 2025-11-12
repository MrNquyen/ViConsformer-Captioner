
import torch
import torch.nn.functional as F
from torch import nn
from typing import List
import math
from icecream import ic

from utils.module_utils import _batch_gather, _get_causal_mask
from utils.registry import registry
from utils.utils import load_vocab
from projects.modules.multimodal_embedding import WordEmbedding, ObjEmbedding, OCREmbedding
from transformers import RobertaPreTrainedModel, RobertaConfig

# -- Previous Embedding
class PrevEmbedding(nn.Module):
    def __init__(self, hidden_size, mutimodal_transformer_config):
        super().__init__()
        self.DEC_LENGTH = mutimodal_transformer_config["max_length"] # Max caption output length
        self.TYPE_NUM = 2 # OCR or FROM_VOCAB
        self.hidden_size = hidden_size

        self.positional_embedding = nn.Embedding(
            num_embeddings=self.DEC_LENGTH, 
            embedding_dim=hidden_size
        )
        self.token_type_embedding = nn.Embedding(
            num_embeddings=self.TYPE_NUM, 
            embedding_dim=hidden_size
        )
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)
        self.Dropout = nn.Dropout(mutimodal_transformer_config["dropout"])


    def init_pe_weights(self):
        """
            Init weight for self.positional_embedding
        """
        # Create positional encoding sin cos from Attention Is All You Need! paper
        pe = torch.zeros(self.DEC_LENGTH, self.hidden_size)
        position = torch.arange(0, self.DEC_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2).float() *
            (-math.log(10000.0) / self.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Assign custom weights
        self.positional_embedding.weight.data = pe.clone()


    def forward(
            self,
            common_vocab_embed,
            ocr_embed,
            prev_inds
        ):
        """
            :params common_vocab_embed    :  common_vocab_len, hidden_size:   All embedding of common vocab
            :params ocr_embed           :  BS, num_ocr, hidden_size     :   All ocr embedding
            :params prev_inds                :  BS, list_prev_idx_in_vocab   :   All idx in vocab of prev word in 
            ----
            Note:
                - Idx of ocr token start from: vocab_len + 1
                (Because append ocr_vocab to common_vocab)
                - When training, input all the gt caption and mask, so the model cannot see the future
            ----
            Function:
                - Lookup table embed position, and get prev embeded vector
        """
        # -- Params
        batch_size = prev_inds.shape[0]
        current_seq_length = prev_inds.shape[1]
        vocab_size = common_vocab_embed.shape[0]
        ocr_size = ocr_embed.shape[1]

        # -- Get prev vector embed
        common_vocab_embed = self.LayerNorm(common_vocab_embed)
        ocr_embed = self.LayerNorm(ocr_embed)
        assert common_vocab_embed.size(-1) == ocr_embed.size(-1)

        common_vocab_embed = common_vocab_embed.unsqueeze(0).expand(batch_size, -1, -1)
        # ic(common_vocab_embed.shape, ocr_embed.shape)
        look_up_table_embedding = torch.concat(
            [common_vocab_embed, ocr_embed],
            dim=1
        )

        last_word_embeddings = _batch_gather(
            x=look_up_table_embedding, 
            inds=prev_inds
        )

        # -- Position 
        position_ids = torch.arange(
            current_seq_length,
            dtype=torch.long,
            device=ocr_embed.device
        )
        # ic(position_ids)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.positional_embedding(position_ids)

        # -- Type embedding: 0: common tokens (False) - 1: ocr_tokens (True)
        type_ids = position_ids.ge(vocab_size).long()
        token_type_embedddings = self.token_type_embedding(type_ids)

        # -- Position and token type
        pos_type_embeddings = position_embeddings + token_type_embedddings 
        # ic(pos_type_embeddings.shape)
        pos_type_embeddings = self.emb_layer_norm(pos_type_embeddings)
        pos_type_embeddings = self.emb_dropout(pos_type_embeddings)

        # -- LastWord, Position, token type
        prev_emb = last_word_embeddings + pos_type_embeddings
        return prev_emb # BS, num_prev_words, hidden_size



# ---------- Encoder as Decoder
class EncoderAsDecoder(RobertaPreTrainedModel):
    def __init__(self, pretrained_model, roberta_config, **kwargs):
        """
            Parameters:
            ----------
                **kwargs:
                    - num_vocab

        """
        super().__init__(roberta_config)
        self.config = registry.get_config("model_attributes")
        self.mmt_config = registry.get_config("model_attributes")["mutimodal_transformer"]
        self.pretrained_model = pretrained_model
        self.encoder = self.pretrained_model.encoder
        self.hidden_size = self.config["hidden_size"]
        self.kwargs = kwargs
        # Build
        self.build()

        # Layer
        fastext_dim = 300
        self.ocr_semantic_linear = nn.Linear(
            in_features=fastext_dim,
            out_features=self.hidden_size
        )
    
    def build(self):
        # self.load_pretrained()
        self.build_layers()


    def build_layers(self):
        # Prev Embedding
        self.prev_embedding = PrevEmbedding(
            hidden_size=self.hidden_size,
            mutimodal_transformer_config=self.mmt_config
        )


    def forward(
            self,
            obj_embed,
            obj_mask,
            ocr_embed,
            ocr_mask,
            semantic_ocr_tokens_feat,
            visual_object_concept_feat,
            common_vocab_embed,
            prev_inds
        ):
        """
            Forward batch through a model
            Parameters:
            ----------
                obj_embed: Tensor((BS, num_obj, hidden_size)) 
                    Obj features embedding 
                obj_mask: 
                    Mask of obj objects
                ocr_embed: Tensor((BS, num_ocr, hidden_size))
                    Ocr features embedding
                ocr_mask: Tensor((BS, num_ocr))
                    Mask of ocr tokens
                semantic_ocr_tokens_feat: Tensor((BS, num_ocr, 300))
                    Output of SgAM module
                visual_object_concept_feat: Tensor((BS, topK, hidden_size))
                    Output of SgAM module
                common_vocab_embed: Tensor((len_vocab, embed_dim))
                    Embed vectors of vocabs
                prev_inds: 
                    Previous inds
        """
        #-- Decoder Embedding
            #~ prev_embed is the lookup embedding of previous words
            #~~%% Training: prev_embed is all the word in the captions
            #~~%% Testing: prev_embed is only the previous word
        prev_embed = self.prev_embedding(
            common_voc_embed=common_vocab_embed,
            ocr_embed=ocr_embed,
            prev_inds=prev_inds
        )

        dec_mask = torch.zeros(
            prev_embed.size(0),
            prev_embed.size(1),
            dtype=torch.float32,
            device=prev_embed.device
        )

        # Convert fasttext dim to hidden state dim
        semantic_ocr_tokens_feat = self.ocr_semantic_linear(semantic_ocr_tokens_feat)

        # -- Encoder for stage t
        # ic(obj_embed.shape, ocr_embed.shape, semantic_ocr_tokens_feat.shape, visual_object_concept_feat.shape, prev_embed.shape)
        encoder_inputs = torch.cat(
            [obj_embed, ocr_embed, semantic_ocr_tokens_feat, visual_object_concept_feat, prev_embed],
            dim=1
        )

        # -- Attention_masks
            #~ ocr_semantic_mask = ocr_mask
            #~ Shape: BS, num_obj + num_ocr * 2 + k + max_length
        visual_concept_mask = torch.ones(
            visual_object_concept_feat.size(0),
            visual_object_concept_feat.size(1),
            dtype=torch.float32,
            device=visual_object_concept_feat.device
        )
        # ic(obj_mask.device, ocr_mask.device, ocr_mask.device, visual_concept_mask.device, dec_mask.device)
        attention_mask = torch.cat(
            [obj_mask, ocr_mask, ocr_mask, visual_concept_mask, dec_mask],
            dim=1
        )

        #-- Offsets of each modality in the joint embedding space
        encoder_input_begin = 0
        ocr_begin = encoder_input_begin + obj_embed.size(1)
        ocr_end = ocr_begin + ocr_embed.size(1)
        encoder_input_embed_end = ocr_end + semantic_ocr_tokens_feat.size(1) + visual_object_concept_feat.size(1)
        dec_input_begin = encoder_input_embed_end
        dec_input_end = dec_input_begin + prev_embed.size(1)
        # ic(dec_input_begin)

        #-- Multihead broadcasting
        end_when_reach_maxlen = attention_mask.size(1) # 
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, end_when_reach_maxlen, 1
        )

        #-- Casual masking for multihead broadcasting
            #~ Start Casual masking at start of dec_input 
        extended_attention_mask[:, :, dec_input_begin:, dec_input_begin:] = \
            _get_causal_mask(dec_input_end - dec_input_begin, encoder_inputs.device)
        
        #-- Valid attention has value 0 - invalid -inf
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        #-- Disable grad
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config["num_layers"]

        # -- Pass forward encoder
            #~ encoder_outputs is a tuple (or BaseModelOutput):
            #~~%%   encoder_outputs[0] == last_hidden_states
            #~~%%   encoder_outputs[1] == (optional) all_hidden_states
            #~~%%   encoder_outputs[2] == (optional) all_attentions
        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, dec_input_begin:]

        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_ocr_output': mmt_ocr_output,
            'mmt_dec_output': mmt_dec_output,
        }
        return results
        
        
