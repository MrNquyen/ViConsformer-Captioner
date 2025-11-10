import torch
import warnings
import os

import numpy as np
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from icecream import ic
from torch.optim.lr_scheduler import LambdaLR

from datasets.dataset import get_loader
from projects.models.device_model import DEVICE
from utils.configs import Config
from utils.model_utils import get_optimizer_parameters
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
        self.model = DEVICE(
            config=self.config.config_model, 
            device=self.device, 
            writer=self.writer # kwargs
        ).to(self.device)


    def build_training_params(self):
        self.max_epochs = self.config.config_training["epochs"]
        self.batch_size = self.config.config_training["batch_size"]
        self.max_iterations = self.config.config_training["max_iterations"]
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
            config_lr_scheduler=self.config.config_lr_scheduler
        )

        # Resume training
        if self.args.resume_file != None:
            self.load_model(self.args.resume_file)


    def build_scheduler(self, optimizer, config_lr_scheduler):
        if not config_lr_scheduler["status"]:
            return None
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config_lr_scheduler["step_size"],
            gamma=config_lr_scheduler["gamma"]
        )
        return lr_scheduler


    def build_loss(self):
        loss_fn = nn.CrossEntropyLoss()
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
        scores_output = self.model(batch)
        return scores_output


    def _extract_loss(self, scores, targets):
        B, max_length, C = scores.shape
        scores_flat = scores.view(B * max_length, C)           # [2*30, 6048]
        targets_flat = targets.view(B * max_length) 
        loss_output = self.loss_fn(
            scores_flat, targets_flat
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
    

    def _run_scheduler(self):
        """
            Learning rate scheduler
        """
        # self.lr_scheduler.step(self.current_iteration)
        self.lr_scheduler.step()


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

        return batch


    #---- MODE
    def train(self):
        self.writer.LOG_INFO("=== Model ===")
        self.writer.LOG_INFO(self.model)

        self.writer.LOG_INFO("Starting training...")
        self.model.train()

        while self.current_iteration < self.max_iterations:
            self.current_epoch += 1
            self.writer.LOG_INFO(f"Training epoch: {self.current_epoch}")
            for batch_id, batch in tqdm(enumerate(self.train_loader), desc="Iterating through train loader"):
                self.writer.LOG_INFO(f"Training batch: {batch_id + 1}")
                batch = self.preprocess_batch(batch)
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
                
                if self.current_iteration < self.max_iterations:
                    break
            
            if self.current_epoch % 2 == 0:
                _, _, val_final_scores, loss = self.evaluate(epoch_id=self.current_epoch, split="val")
                # _, _, final_scores = self.evaluate(epoch_id=self.current_epoch, split="test")
                if val_final_scores > best_scores:
                    best_scores = val_final_scores
                    self.save_model(
                        model=self.model,
                        loss=loss,
                        optimizer=self.optimizer,
                        epoch=self.current_epoch, 
                        metric_score=best_scores
                    )
    
    def evaluate(self, epoch_id=None, split="val"):
        if hasattr(self, f"{split}_loader"):
            dataloader = getattr(self, f"{split}_loader")
        else:
            self.writer_evaluation.LOG_ERROR(f"No dataloader for {split} split")
            raise ModuleNotFoundError
        
        with torch.inference_mode():
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
                
                #~ Calculate loss
                scores_output, pred_inds = self._forward_pass(batch)
                targets = self.model.word_embedding.get_prev_inds(
                    list_captions, list_ocr_tokens
                ).to(self.device)
                loss = self._extract_loss(scores_output, targets)
                losses.append(loss)
                #~ Metrics calculation
                if not epoch_id==None:
                    self.writer_evaluation.LOG_INFO(f"Logging at epoch {epoch_id}")
                
                pred_caps = self.get_pred_captions(scores_output, list_ocr_tokens)
                for id, pred_cap, ref_cap in zip(list_id, pred_caps, list_captions):
                    hypo[id] = [pred_cap]
                    ref[id]  = [ref_cap]
            
            # Calculate Metrics
            final_scores = metric_calculate(ref, hypo)
            loss = sum(losses) / len(losses) 
            self.writer_evaluation.LOG_INFO(f"|| Metrics Calculation || {split} split || epoch: {epoch_id} || loss: {loss}")
            self.writer_evaluation.LOG_INFO(f"Final scores:\n{final_scores}")
            
            # Turn on train mode to continue training
            self.model.train()
        return hypo, ref, final_scores, loss
    

    def inference(self, mode, save_dir):
        """
            Parameters:
                mode:   Model to run "val" or "test"
        """
        if mode=="val":
            self.writer_inference.LOG_INFO("=== Inference Validation Split ===")
            hypo, ref = self.evaluate(epoch_id="Inference val set", split="val")
        elif mode=="test":
            self.writer_inference.LOG_INFO("=== Inference Test Split ===")
            hypo, ref = self.evaluate(epoch_id="Inference test set", split="test")
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
    def save_model(self, model, loss, optimizer, epoch, metric_score):
        if os.path.exists(self.args.save_dir):
            self.writer("Save dir not exist")
            raise FileNotFoundError
        
        model_path = os.path.join(self.args.save_dir, f"model_{epoch}.pth")
        self.writer.LOG_DEBUG(f"Model save at {model_path}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            "metric_score": metric_score
        }, model_path)

    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.writer.LOG_INFO(f"=== Load model at epoch: {epoch} || loss: {loss} ===")

    #---- METRIC CALCULATION
    def get_pred_captions(self, scores, ocr_tokens):
        """
            Predict batch
        """
        # Get logits
        pred_logits = F.softmax(scores, dim=-1)
        pred_ids = np.argmax(pred_logits, axis=1)

        # Captioning
        common_vocab = self.model.word_embedding.common_vocab
        vocab_size = common_vocab.get_size()
        ocr_vocab_object = OCRVocab(ocr_tokens=ocr_tokens)
        captions_pred = [
            " ".join([
                common_vocab.get_idx_word(idx)
                if idx < vocab_size
                else ocr_vocab_object[i].get_idx_word(idx - vocab_size)
                for idx in item_pred_ids
            ])
            for i, item_pred_ids in enumerate(pred_ids)
        ]
        return captions_pred # BS, 
    

    def save_inference(self, hypo, ref, save_dir, name=""):
        save_path = os.path.join(save_dir, f"{name}_reference")
        inference: dict = {
            id : {
                "gt": ref[id],
                "pred": hypo[id]
            } for id in ref.keys()
        }
        save_json(save_path, content=inference)