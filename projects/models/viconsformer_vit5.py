import torch
import fasttext
import math
import numpy as np
from torch import nn
from PIL import Image
from icecream import ic
from torch.nn import functional as F
from projects.vit5_modules_new.multimodal_embedding import OBJEmbedding, OCREmbedding, Sync
from projects.vit5_modules_new.encoder import ViConsformerEncoder 
from projects.vit5_modules_new.decoder_mmt import Decoder 
from projects.vit5_modules_new.classifier import Classifier 
from utils.registry import registry
from utils.module_utils import _batch_padding, _batch_padding_string
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5ForConditionalGeneration

#---------- MODEL ----------
class BaseModel(nn.Module):
    def __init__(self):
        """
            :params config: Model Config
            :params device: device cuda
        """
        super().__init__()
        self.config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")

        self.hidden_size = self.config["hidden_size"]
        self.build_model_init()


    def build_model_init(self):
        # Finetune module is the module has lower lr than others module
        self.finetune_modules = []

    def add_finetune_modules(self, module: nn.Module):
        self.finetune_modules.append({
            'module': module,
            'lr_scale': self.config["adjust_optimizer"]["lr_scale"],
        })



#---------- DEVICE MODEL ----------
class ViConsformer(BaseModel):
    def __init__(self):
        """
            :params config: Model Config
            :params device: device cuda
        """
        super().__init__()
        self.build()


    #---- BUILD
    def build(self):
        self.writer.LOG_INFO("=== Build model params ===")
        self.build_model_params()
        
        self.writer.LOG_INFO("=== Build model pretrained ===")
        self.load_pretrained()

        self.writer.LOG_INFO("=== Build model layers ===")
        self.build_layers()

        self.writer.LOG_INFO("=== Build model sync ===")
        self.build_sync()

        self.writer.LOG_INFO("=== Build Adjust learning rate ===")
        self.adjust_lr()


    def build_model_params(self):
        self.ocr_config = self.config["ocr"]
        self.obj_config = self.config["obj"]
        self.num_ocr, self.num_obj = self.ocr_config["num_ocr"], self.obj_config["num_obj"]
        self.dim_ocr, self.dim_obj = self.ocr_config["dim"], self.obj_config["dim"]
        self.feature_dim = self.config["feature_dim"]
        self.hidden_size = self.config["hidden_size"]
        self.max_dec_length = self.config["mutimodal_transformer"]["max_length"]


    def load_pretrained(self):
        self.model_name = self.config["pretrained"]
        ic(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)

        #-- Load pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            config=config
        ).to(self.device)

        #-- Load Encoder, Decoder, Classifier
        self.model.gradient_checkpointing_enable()
        self.model_encoder = self.model.encoder
        self.model_decoder = self.model.decoder
        self.model_classifier = self.model.lm_head

        self.encoder_embed_tokens_layer = self.model_encoder.embed_tokens
        self.encoder_block_layer = self.model_encoder.block


    def build_sync(self):
        self.sync_ocr = Sync(
            in_dim=self.dim_ocr,
            out_dim=self.hidden_size
        )

        self.sync_obj = Sync(
            in_dim=self.dim_obj,
            out_dim=self.hidden_size
        )

    def build_layers(self):
        self.encoder = ViConsformerEncoder(
            tokenizer=self.tokenizer,
            encoder_embed_tokens_layer=self.encoder_embed_tokens_layer,
            encoder_block_layer=self.encoder_block_layer
        )
        self.decoder = Decoder(self.model_decoder)
        self.classifier = Classifier(self.model_classifier)


    #---- ADJUST LR FOR SPECIFIC MODULES
    def adjust_lr(self):
        pass

    #---- HELPER FUNCTION
    def preprocess_batch(self, batch):
        """
            Function:
                - Padding ocr and obj to the same length
                - Create mask for ocr and obj
        """
        box_pad = torch.rand((1, 4))
        # Padding ocr
        ocr_feat_pad = torch.rand((1, self.dim_ocr))
        batch["list_ocr_boxes"], ocr_mask = _batch_padding(batch["list_ocr_boxes"], max_length=self.num_ocr, pad_value=box_pad)
        batch["list_ocr_feat"] = _batch_padding(batch["list_ocr_feat"], max_length=self.num_ocr, pad_value=ocr_feat_pad, return_mask=False)
        
        batch["list_ocr_tokens"] = _batch_padding_string(batch["list_ocr_tokens"], max_length=self.num_ocr, pad_value="<pad>", return_mask=False)
        batch["ocr_mask"] = ocr_mask

        # Padding obj
        obj_feat_pad = torch.rand((1, self.dim_obj))
        batch["list_obj_boxes"], obj_mask = _batch_padding(batch["list_obj_boxes"], max_length=self.num_obj, pad_value=box_pad)
        batch["list_obj_feat"] = _batch_padding(batch["list_obj_feat"], max_length=self.num_obj, pad_value=obj_feat_pad, return_mask=False)
        batch["obj_mask"] = obj_mask

        # Padding ocr scores/ confidence
        ocr_score_pad = 0
        batch["list_ocr_scores"] = _batch_padding_string(batch["list_ocr_scores"], max_length=self.num_ocr, pad_value=ocr_score_pad, return_mask=False)

        return batch

    def map_device(self, batch):
        batch["list_ocr_feat"] = batch["list_ocr_feat"].to(self.device)
        batch["list_obj_feat"] = batch["list_obj_feat"].to(self.device)
        batch["list_obj_boxes"] = batch["list_obj_boxes"].to(self.device)
        batch["list_ocr_boxes"] = batch["list_obj_boxes"].to(self.device)
        batch["ocr_mask"] = batch["ocr_mask"].to(self.device)
        batch["obj_mask"] = batch["obj_mask"].to(self.device)
        return batch

    def sync_ocr_obj(self, batch):
        """
            Sync OCR and OBJ features to the same feat_dim
        """
        batch["list_ocr_feat"] = self.sync_ocr(batch["list_ocr_feat"]).to(self.device)
        batch["list_obj_feat"] = self.sync_obj(batch["list_obj_feat"]).to(self.device)
        return batch
    

    def get_optimizer_parameters(self, config_optimizer):
        """
            --------
            Function:
                - Modify learning rate
                - Fine-tuning layer has lower learning rate than others
        """
        optimizer_param_groups = []
        base_lr = config_optimizer["params"]["lr"]
        scale_lr = config_optimizer["lr_scale"]
        base_lr = float(base_lr)
        
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * scale_lr
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})
        
        # check_overlap(finetune_params_set, remaining_params)
        # self.writer.LOG_INFO(f"%=== Optimer params groups ===% \n {optimizer_param_groups}")
        return optimizer_param_groups


    def forward_mmt(
        self,
        ocr_embed,
        obj_embed,
        ocr_mask,
        obj_mask,
        ocr_tokens_embed_vit5,
        ocr_tokens_attention_mask_vit5,
        semantic_ocr_tokens_feat,
        visual_object_concept_feat,
        decoder_input_ids,
        decoder_attention_mask

    ):
        """
            Forward batch to model
        """
        # -- Decoder
        mmt_results: dict  = self.decoder(
            obj_embed=obj_embed,
            obj_mask=obj_mask,
            ocr_embed=ocr_embed,
            ocr_mask=ocr_mask,
            ocr_tokens_embed_vit5=ocr_tokens_embed_vit5,
            ocr_tokens_attention_mask_vit5=ocr_tokens_attention_mask_vit5,
            semantic_ocr_tokens_feat=semantic_ocr_tokens_feat,
            visual_object_concept_feat=visual_object_concept_feat,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        return mmt_results
    

    def forward_output(self, results):
        mmt_dec_output = results['vit5_dec_last_hidden_state'] # BS, max_length, hidden_size

        fixed_scores = self.classifier(mmt_dec_output) #  BS, max_length, num_vocab
        results['scores'] = fixed_scores
        return fixed_scores


    def forward(self, batch):
        batch = self.map_device(batch)
        batch = self.sync_ocr_obj(batch)

        #-- Forward to layer
        encoder_outputs = self.encoder(batch)
        encoder_output_embed = encoder_outputs.last_hidden_state
        encoder_output_mask = encoder_outputs.attention_mask

        caption_inputs = self.encoder_caption.tokenize(batch["list_captions"])
        caption_input_ids = caption_inputs["input_ids"]
        caption_attention_mask = caption_inputs["attention_mask"]

        #~ Shift for decoder
        batch_size = ocr_tokens_embed_vit5.size(0)
        vocab_size = self.classifier.get_vocab_size()

            #~: Labels:     Tôi  là  AI  .  <EOS>
        labels_input_ids = caption_input_ids.clone()
        labels_input_ids[labels_input_ids == self.encoder_caption.get_pad_token_id()] = -100

        #-- Training and Inference
        if self.training:
            #~ Decoder input: shift right (prepend pad_token, remove last token)
            #~ Example: [Tôi, là, AI, ., <eos>] -> [<pad>, Tôi, là, AI, .]
            shift_decoder_input_ids = self.decoder._shift_right(caption_input_ids.clone())
            decoder_attention_mask = (shift_decoder_input_ids != self.encoder_caption.get_pad_token_id()).long()

            results = self.forward_mmt(
                ocr_embed=ocr_feat_embed,
                obj_embed=obj_embed_feat,
                ocr_mask=ocr_mask,
                obj_mask=obj_mask,
                ocr_tokens_embed_vit5=ocr_tokens_embed_vit5,
                ocr_tokens_attention_mask_vit5=ocr_tokens_attention_mask_vit5,
                semantic_ocr_tokens_feat=semantic_ocr_tokens_feat,
                visual_object_concept_feat=visual_object_concept_feat,
                decoder_input_ids=shift_decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask

            )
            scores = self.forward_output(results)
            return scores, caption_input_ids, labels_input_ids
        else:
            #~ Greedy Search
            eos_id = self.encoder_caption.get_eos_token_id()
            pad_id = self.encoder_caption.get_pad_token_id()

            with torch.no_grad():
                scores = torch.zeros((batch_size, self.max_dec_length, vocab_size), device=self.device)
                decoder_input_ids = torch.full(
                    (batch_size, 1),
                    fill_value=pad_id,
                    dtype=torch.long,
                    device=self.device
                )

                #~ Iterate through max decoder length
                for step in range(self.max_dec_length):
                    decoder_attention_mask = (decoder_input_ids != pad_id).long()
                    results = self.forward_mmt(
                        ocr_embed=ocr_feat_embed,
                        obj_embed=obj_embed_feat,
                        ocr_mask=ocr_mask,
                        obj_mask=obj_mask,
                        ocr_tokens_embed_vit5=ocr_tokens_embed_vit5,
                        ocr_tokens_attention_mask_vit5=ocr_tokens_attention_mask_vit5,
                        semantic_ocr_tokens_feat=semantic_ocr_tokens_feat,
                        visual_object_concept_feat=visual_object_concept_feat,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask
                    )
                    step_scores = self.forward_output(results=results)
                    argmax_inds = step_scores.argmax(dim=-1).unsqueeze(-1)

                    #~ Assign
                    scores[:, step, :] = step_scores[:, -1, :]
                    decoder_input_ids = torch.concat([decoder_input_ids, argmax_inds[:, -1]], dim=1)

                # gen_ids = decoder_input_ids[:, 1:] #-- Ignore the first pad token
                return scores, decoder_input_ids, labels_input_ids