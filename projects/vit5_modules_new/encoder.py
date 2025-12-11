import torch
from icecream import ic
from dataclasses import dataclass
from typing import Optional, List
from torch import nn
from projects.vit5_modules_new.attention import SpartialCirclePosition
from projects.vit5_modules_new.scene_text_embedding import SceneTextEmbedding
from projects.vit5_modules_new.tokenizer import WordTokenizer
from projects.vit5_modules_new.multimodal_embedding import OCREncoder, OBJEncoder
# from transformers.cache_utils import DynamicCache, Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import ModelOutput
from transformers.models.t5.modeling_t5 import *
from transformers.models.t5.modeling_t5 import (
    Cache,
    T5Stack,
    DynamicCache,
    EncoderDecoderCache,
    BaseModelOutputWithPastAndCrossAttentions,

)
from utils.registry import registry

@dataclass
class CustomBaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    attention_mask: Optional[torch.FloatTensor] = None


# From https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
class ViConsformerEncoder(T5Stack):
    def __init__(
        self,
        config,
        word_tokenizer,
        encoder_embed_tokens_layer,
        encoder_block_layer
    ):
        super(ViConsformerEncoder, self).__init__(config)
        self.writer = registry.get_writer("common")
        self.model_config = registry.get_config("model_attributes")
        # self.device = registry.get_args("device")
        self.hidden_size = self.model_config["hidden_size"]
        self.max_length = self.model_config["max_length"]
        self.build_config()

        self.word_tokenizer = word_tokenizer
        self.embed_tokens = encoder_embed_tokens_layer
        self.block = encoder_block_layer

        self.ocr_encoder = OCREncoder()
        self.obj_encoder = OBJEncoder()
        self.scene_text_embedding = SceneTextEmbedding()
        self.spatial_circle_position = SpartialCirclePosition()


    def build_config(self):
        self.ocr_config = self.model_config["ocr"]
        self.obj_config = self.model_config["obj"]

    
    def forward(
        self, 
        batch,
        input_ids=None, # Question
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,   
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        #-- Custom setup
        list_obj_feat   = batch["list_obj_feat"]
        list_ocr_feat   = batch["list_ocr_feat"]
        list_ocr_boxes  = batch["list_ocr_boxes"]
        list_obj_boxes  = batch["list_obj_boxes"]
        list_questions  = batch["list_questions"]
        list_ocr_tokens = batch["list_ocr_tokens"]
        ocr_mask        = batch["ocr_mask"]
        obj_mask        = batch["obj_mask"]


        #~ OCR Embedding
        list_ocr_tokens_string  = [" <context> ".join(ocr_tokens) for ocr_tokens in batch["list_ocr_tokens"]]
        ocr_tokens_inputs       = self.word_tokenizer.tokenize(list_ocr_tokens_string, max_length=self.ocr_config["num_ocr"])
        ocr_tokens_attention_mask   = ocr_tokens_inputs["attention_mask"]
        ocr_tokens_input_ids        = ocr_tokens_inputs["input_ids"]
        ocr_tokens_embed            = self.embed_tokens(ocr_tokens_input_ids)
        
        ocr_features = self.scene_text_embedding(
            batch=batch, 
            ocr_tokens_embed_vit5=ocr_tokens_embed
        )
        ocr_features, stack_neibor_attn = self.ocr_encoder(
            ocr_features=ocr_features, 
            ocr_mask=ocr_tokens_attention_mask #-- Using ViT5 inputs attention mask
        )
        ocr_spatial_position_embed = self.spatial_circle_position(
            batch=batch, 
            features=ocr_features, 
            list_boxes=list_ocr_boxes, 
            features_mask=ocr_tokens_attention_mask
        )

        #~ OBJ Embedding
        obj_features = self.obj_encoder(batch)
        obj_spatial_position_embed = self.spatial_circle_position(
            batch=batch, 
            features=obj_features, 
            list_boxes=list_obj_boxes, 
            features_mask=obj_mask
        )

        #~ Question Embedding
        if input_ids is None:
            inputs = self.word_tokenizer.tokenize(list_questions, to_batch_max_length=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        question_features = self.embed_tokens(input_ids)

        #~ Fusion to input to encoder
        inputs_embeds = torch.concat([ #-- Follow transformer vit5 source code
            question_features,
            ocr_spatial_position_embed,
            obj_spatial_position_embed,
        ], dim=1)

        attention_mask = torch.concat([ #-- Follow transformer vit5 source code
            attention_mask,
            ocr_tokens_attention_mask,
            obj_mask,
        ], dim=1)

            
        #-- Modeling_t5 transformer setup
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if input_ids is not None and inputs_embeds is not None:
        #     err_msg_prefix = "decoder_" if self.is_decoder else ""
        #     raise ValueError(
        #         f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        #     )
        # if input_ids is not None:
        #     input_shape = input_ids.size()
        #     input_ids = input_ids.view(-1, input_shape[-1])
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config), DynamicCache(config=self.config)
                    )
                else:
                    past_key_values = DynamicCache(config=self.config)
        elif not self.is_decoder:
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if self.config.is_decoder:
            attention_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values,
                output_attentions,
            )
        elif attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            attention_mask = None

        encoder_extended_attention_mask = None
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_extended_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for layer_module in self.block:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return CustomBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask.squeeze(1).squeeze(1)
        )

    