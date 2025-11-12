import torch
import warnings
import os

import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from icecream import ic
from torch.optim.lr_scheduler import LambdaLR

from projects.datasets.dataset import get_loader
from projects.models.device_model import DEVICE
from utils.configs import Config
from utils.model_utils import get_optimizer_parameters, lr_lambda_update
from utils.module_utils import _batch_padding, _batch_padding_string
from utils.logger import Logger
from utils.vocab import OCRVocab
from utils.metrics import metric_calculate
from utils.registry import registry
from utils.utils import save_json, count_nan, check_requires_grad, set_seed


# ~Trainer~
set_seed(2021)
class Trainer():
    def __init__(self, config, args):
        self.args = args
        self.device = args.device
        self.config = Config(config)

        print("Build Logger")
        self.writer = Logger(name="all")
        self.writer_evaluation = Logger(name="evaluation")
        self.writer_inference = Logger(name="inference")
        self.build()


    #---- LOAD TASK
    def load_task(self):
        batch_size = self.config.config_training["batch_size"]
        self.train_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="train")
        self.val_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="val")
        self.test_loader = get_loader(dataset_config=self.config.config_dataset, batch_size=batch_size, split="test")


    #---- REGISTER
    def build_registry(self):
        # Build writer
        registry.set_module("writer", name="common", instance=self.writer)
        registry.set_module("writer", name="evaluation", instance=self.writer_evaluation)
        registry.set_module("writer", name="inference", instance=self.writer_inference)
        
        # Build args
        registry.set_module("args", name=None, instance=self.args)

        # Build config
        self.config.build_registry()


    #---- BUILD
    def build(self):
        self.writer.LOG_INFO("=== START BUILDING ===")
        self.build_registry()
        self.build_model()
        self.load_task()
        self.build_training_params()


    def build_model(self):
        self.model = DEVICE()
        self.model = self.model.to(self.device)


    def build_training_params(self):
        self.max_epochs = self.config.config_training["epochs"]
        self.batch_size = self.config.config_training["batch_size"]
        self.max_iterations = self.config.config_training["max_iterations"]
        self.snapshot_interval = self.config.config_training["snapshot_interval"]
        self.early_stop_patience = self.config.config_training["early_stopping"]["patience"]
        self.current_iteration = 0
        self.current_epoch = 0

        # Training
        self.optimizer = self.build_optimizer(
            model=self.model,
            config_optimizer=self.config.config_optimizer
        )
        self.loss_fn = self.build_loss()
        self.lr_scheduler = self.build_scheduler(
            optimizer=self.optimizer,
            config_lr_scheduler=self.config.config_training["lr_scheduler"]
        )

        # Resume training
        if self.args.resume_file != None:
            self.load_model(self.args.resume_file)


    def build_scheduler(self, optimizer, config_lr_scheduler):
        if not config_lr_scheduler["status"]:
            return None

        scheduler_func = lambda x: lr_lambda_update(x, config_lr_scheduler)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=scheduler_func
        )
        return lr_scheduler


    def build_loss(self):
        pad_idx = self.model.word_embedding.common_vocab.get_pad_index()
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        return loss_fn


    def build_optimizer(self, model, config_optimizer):
        if "type" not in config_optimizer:
            raise ValueError(
                "Optimizer attributes must have a 'type' key "
                "specifying the type of optimizer. "
                "(Custom or PyTorch)"
            )
        optimizer_type = config_optimizer["type"]

        #-- Load params
        if not hasattr(config_optimizer, "params"):
            warnings.warn(
                "optimizer attributes has no params defined, defaulting to {}."
            )
        optimizer_params = getattr(config_optimizer, "params", {})
        
        #-- Load optimizer class
        if not hasattr(torch.optim, optimizer_type):
            raise ValueError(
                "No optimizer found in torch.optim"
            )
        optimizer_class = getattr(torch.optim, optimizer_type)
        parameters = get_optimizer_parameters(
            model=model, 
            config=config_optimizer
        )
        optimizer = optimizer_class(parameters, **optimizer_params)
        return optimizer
    

    #---- STEP
    def _forward_pass(self, batch):
        """
            Forward to model
        """
        scores_output, prev_inds = self.model(batch)
        return scores_output, prev_inds


    def _extract_loss(self, scores, targets):
        """
            scores.shape: torch.Size([B, max_length, num_vocab])
            targets.shape: torch.Size([B, max_length])
        """
        num_vocab = scores.size(-1)
        scores_reshape = scores[:, 1:, :].reshape(-1, num_vocab)
        targets_reshape = targets[:, 1:].reshape(-1)
        loss_output = self.loss_fn(
            # scores, targets
            scores_reshape, targets_reshape
        ) 
        return loss_output


    def _backward(self, loss):
        """
            Backpropagation
        """
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self._run_scheduler()
    

    def _gradient_clipping(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


    def _run_scheduler(self):
        """
            Learning rate scheduler
        """
        self.lr_scheduler.step()
        if self.current_iteration % 1000 == 0:
            lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            self.writer.LOG_INFO(f"Iteration {self.current_iteration} LRs: {lrs}")


    def _backward(self, loss):
        """
            Backpropagation
        """
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self._run_scheduler()


    #---- MODE
    def preprocess_batch(self, batch):
        """
            Function:
                - Padding ocr and obj to the same length
                - Create mask for ocr and obj
        """
        model_config = self.model.config
        dim_ocr = model_config["ocr"]["dim"]
        dim_obj = model_config["obj"]["dim"]
        box_pad = torch.rand((1, 4))

        # Padding ocr
        ocr_feat_pad = torch.rand((1, dim_ocr))
        batch["list_ocr_boxes"], ocr_mask = _batch_padding(batch["list_ocr_boxes"], max_length=model_config["ocr"]["num_ocr"], pad_value=box_pad)
        batch["list_ocr_feat"] = _batch_padding(batch["list_ocr_feat"], max_length=model_config["ocr"]["num_ocr"], pad_value=ocr_feat_pad, return_mask=False)
        batch["list_ocr_tokens"] = _batch_padding_string(batch["list_ocr_tokens"], max_length=model_config["ocr"]["num_ocr"], pad_value="<pad>", return_mask=False)
        batch["ocr_mask"] = ocr_mask

        # Padding obj
        obj_feat_pad = torch.rand((1, dim_obj))
        batch["list_obj_boxes"], obj_mask = _batch_padding(batch["list_obj_boxes"], max_length=model_config["obj"]["num_obj"], pad_value=box_pad)
        batch["list_obj_feat"] = _batch_padding(batch["list_obj_feat"], max_length=model_config["obj"]["num_obj"], pad_value=obj_feat_pad, return_mask=False)
        batch["obj_mask"] = obj_mask

        # Padding ocr scores/ confidence
        ocr_score_pad = 0
        batch["list_ocr_scores"] = _batch_padding_string(batch["list_ocr_scores"], max_length=model_config["ocr"]["num_ocr"], pad_value=ocr_score_pad, return_mask=False)
        batch["list_ocr_scores"] = torch.tensor(batch["list_ocr_scores"])

        # Depth Features 
        depth_pad = torch.zeros((1, 1))
            #~ For OBJ
        list_obj_depth_feat = batch["list_obj_depth_feat"]
        list_obj_depth_feat = [
            depth_value if len(depth_value) > 0 else [0.1]
            for depth_value in list_obj_depth_feat
        ]
        list_obj_depth_feat = [torch.tensor(depth_feat).unsqueeze(-1) for depth_feat in list_obj_depth_feat]
        batch["list_obj_depth_feat"] = _batch_padding(list_obj_depth_feat, model_config["obj"]["num_obj"], depth_pad, return_mask=False)

            #~ For OCR
        list_ocr_depth_feat = batch["list_ocr_depth_feat"]
        list_ocr_depth_feat = [
            depth_value if len(depth_value) > 0 else [0.1]
            for depth_value in list_ocr_depth_feat
        ]
        list_ocr_depth_feat = [torch.tensor(depth_feat).unsqueeze(-1) for depth_feat in list_ocr_depth_feat]
        batch["list_ocr_depth_feat"] = _batch_padding(list_ocr_depth_feat, model_config["ocr"]["num_ocr"], depth_pad, return_mask=False)

        # CLIP Feat
        list_clip_image_feat = batch["list_clip_image_feat"]
        list_clip_image_feat = torch.tensor(list_clip_image_feat).unsqueeze(1)
        batch["list_clip_image_feat"] = list_clip_image_feat

        return batch
    
    def match_device(self, batch):
        batch["list_ocr_boxes"] = batch["list_ocr_boxes"].to(self.device).to(torch.float)
        batch["list_ocr_feat"] = batch["list_ocr_feat"].to(self.device).to(torch.float)
        batch["ocr_mask"] = batch["ocr_mask"].to(self.device).to(torch.float)
        batch["list_obj_boxes"] = batch["list_obj_boxes"].to(self.device).to(torch.float)
        batch["list_obj_feat"] = batch["list_obj_feat"].to(self.device).to(torch.float)
        batch["obj_mask"] = batch["obj_mask"].to(self.device).to(torch.float)
        batch["list_obj_depth_feat"] = batch["list_obj_depth_feat"].to(self.device).to(torch.float)
        batch["list_ocr_depth_feat"] = batch["list_ocr_depth_feat"].to(self.device).to(torch.float)
        batch["list_clip_image_feat"] = batch["list_clip_image_feat"].to(self.device).to(torch.float)
        batch["list_ocr_scores"] = batch["list_ocr_scores"].to(self.device).to(torch.float)
        return batch


    #---- MODE
    def train(self):
        self.writer.LOG_INFO("=== Model ===")
        self.writer.LOG_INFO(self.model)

        self.writer.LOG_INFO("Starting training...")
        self.model.train()

        best_scores = -1
        while self.current_iteration < self.max_iterations:
            self.current_epoch += 1
            self.writer.LOG_INFO(f"Training epoch: {self.current_epoch}")
            for batch_id, batch in tqdm(enumerate(self.train_loader), desc="Iterating through train loader"):
                self.writer.LOG_INFO(f"Training batch: {batch_id + 1}")
                batch = self.preprocess_batch(batch)
                batch = self.match_device(batch)
                list_ocr_tokens = batch["list_ocr_tokens"]
                list_captions = batch["list_captions"]

                self.current_iteration += 1
                #~ Loss cal
                scores_output, caption_inds = self._forward_pass(batch)
                targets = self.model.word_embedding.get_prev_inds(
                    list_captions, list_ocr_tokens
                ).to(self.device)
                loss = self._extract_loss(scores_output, targets)
                self._backward(loss)
                
                if self.current_iteration > self.max_iterations:
                    break
            
                if self.current_iteration % self.snapshot_interval == 0:
                    _, _, val_final_scores, loss = self.evaluate(epoch_id=self.current_iteration, split="val")
                    # _, _, final_scores = self.evaluate(epoch_id=self.current_epoch, split="test")
                    if val_final_scores["CIDEr"] > best_scores:
                        self.early_stop_counter = 0
                        best_scores = val_final_scores["CIDEr"]
                        self.save_model(
                            model=self.model,
                            loss=loss,
                            optimizer=self.optimizer,
                            lr_scheduler=self.lr_scheduler,
                            epoch=self.current_epoch, 
                            iteration=self.current_iteration,
                            metric_score=best_scores,
                            use_name="best"
                        )
                    else:
                        self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_stop_patience:
                        self.writer.LOG_INFO("Early stopping triggered.")
                        break
                    
                    self.save_model(
                        model=self.model,
                        loss=loss,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        epoch=self.current_epoch, 
                        iteration=self.current_iteration,
                        metric_score=best_scores,
                        use_name=self.current_iteration
                    )
                    self.save_model(
                        model=self.model,
                        loss=loss,
                        optimizer=self.optimizer,
                        lr_scheduler=self.lr_scheduler,
                        epoch=self.current_epoch, 
                        iteration=self.current_iteration,
                        metric_score=best_scores,
                        use_name="last"
                    )
    
    
    def evaluate(self, epoch_id=None, split="val"):
        if hasattr(self, f"{split}_loader"):
            dataloader = getattr(self, f"{split}_loader")
        else:
            self.writer_evaluation.LOG_ERROR(f"No dataloader for {split} split")
            raise ModuleNotFoundError
        
        with torch.no_grad():
            hypo: dict = {}
            ref : dict = {}
            losses = []
            self.model.eval()

            for batch_id, batch in tqdm(enumerate(dataloader), desc=f"Evaluating {split} loader"):
                self.writer_evaluation.LOG_INFO(f"Evaluate batch: {batch_id + 1}")
                list_id = batch["list_id"]
                list_ocr_tokens = batch["list_ocr_tokens"]
                list_captions = batch["list_captions"]
                batch = self.preprocess_batch(batch)
                batch = self.match_device(batch)
                
                #~ Calculate loss
                scores_output, pred_inds = self._forward_pass(batch)
                targets = self.model.word_embedding.get_prev_inds(
                    list_captions, list_ocr_tokens
                ).to(self.device)

                loss = self._extract_loss(scores_output, targets)
                loss_scalar = loss.detach().cpu().item()
                ic(loss_scalar)
                losses.append(loss_scalar)
                
                #~ Metrics calculation
                if not epoch_id==None:
                    self.writer_evaluation.LOG_INFO(f"Logging at epoch {epoch_id}")
                
                pred_caps = self.get_pred_captions(pred_inds, list_ocr_tokens)
                ic(pred_caps)
                for id, pred_cap, ref_cap in zip(list_id, pred_caps, list_captions):
                    hypo[id] = [pred_cap]
                    ref[id]  = [ref_cap]
            
            # Calculate Metrics
            # ic(ref, hypo)
            final_scores = metric_calculate(ref, hypo)
            # ic(losses)
            avg_loss = sum(losses) / len(losses) 
            self.writer_evaluation.LOG_INFO(f"|| Metrics Calculation || {split} split || epoch: {epoch_id} || loss: {avg_loss}")
            self.writer_evaluation.LOG_INFO(f"Final scores:\n{final_scores}")
            
            # Turn on train mode to continue training
            self.model.train()
        return hypo, ref, final_scores, avg_loss



    def inference(self, mode, save_dir):
        """
            Parameters:
                mode:   Model to run "val" or "test"
        """
        hypo, ref = {}, {}
        if mode=="val":
            self.writer_inference.LOG_INFO("=== Inference Validation Split ===")
            hypo, ref, _, _ = self.evaluate(epoch_id="Inference val set", split="val")
        elif mode=="test":
            self.writer_inference.LOG_INFO("=== Inference Test Split ===")
            hypo, ref, _, _ = self.evaluate(epoch_id="Inference test set", split="test")
        else:
            self.writer_inference.LOG_ERROR(f"No mode available for {mode}")
        
        # Save Inference
        self.save_inference(
            hypo=hypo,
            ref=ref,
            save_dir=save_dir,
            name=mode
        )
        return hypo, ref


    #---- FINISH
    def save_model(self, model, loss, optimizer, lr_scheduler, epoch, iteration, metric_score, use_name=""):
        if not os.path.exists(self.args.save_dir):
            self.writer.LOG_INFO("Save dir not exist")
            os.makedirs(self.args.save_dir, exist_ok=True)
        
        model_path = os.path.join(self.args.save_dir, f"checkpoints/model_{use_name}.pth")
        self.writer.LOG_DEBUG(f"Model save at {model_path}")

        torch.save({
            'epoch': epoch,
            "iteration": iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': getattr(lr_scheduler, 'state_dict', lambda: None)(),
            'loss': loss,
            "metric_score": metric_score
        }, model_path)

    
    def load_model(self, model_path):
        # ic(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #### DEBUG
        for param_group in self.optimizer.param_groups:
            ic(param_group['lr'])
        #### DEBUG

        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iteration = checkpoint.get("iteration", 600)
        
        if self.lr_scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            try:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                # fallback: advance scheduler to current_iteration
                self.writer.LOG_INFO("Warning: failed to load scheduler state; advancing scheduler to iteration")
                self.lr_scheduler.step(self.current_iteration)
        else:
            # ensure scheduler is aligned with optimizer param_group's lrs
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(self.current_iteration)

        loss = checkpoint['loss']
        self.writer.LOG_INFO(f"=== Load model at epoch: {self.current_epoch} || loss: {loss} ===")



    #---- METRIC CALCULATION
    def get_pred_captions(self, pred_inds, ocr_tokens):
        """
            Predict batch
        """
        # Captioning
        common_vocab = self.model.word_embedding.common_vocab
        vocab_size = common_vocab.get_size()
        ocr_vocab_object = OCRVocab(ocr_tokens=ocr_tokens)
        captions_pred = []
        for i, item_pred_inds in enumerate(pred_inds):
            get_ocr = ocr_vocab_object[i].get_idx_word
            words = (
                common_vocab.get_idx_word(idx.item())
                if idx < vocab_size
                else get_ocr(idx.item() - vocab_size)
                for idx in item_pred_inds
            )
            captions_pred.append(" ".join(words))
        return captions_pred 
    

    def save_inference(self, hypo, ref, save_dir, name=""):
        save_path = os.path.join(save_dir, f"{name}_reference")
        inference: dict = {
            id : {
                "gt": ref[id],
                "pred": hypo[id],
            } for id in ref.keys()
        }
        save_json(save_path, content=inference)