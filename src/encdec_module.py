from torch import nn as nn
from torch.nn import functional as F
from transformers import *
from utils import *
from pytorch_lightning import seed_everything
from argparse import Namespace

import torch
import pytorch_lightning as pl
import numpy as np
import random
import warnings
warnings.filterwarnings('error')


# 0: path, 1: config, 2: tokenizer, 3: model
model_maps = {
    "bart-base": ["facebook/bart-base", BartConfig, BartTokenizer, BartForConditionalGeneration],
    "bart-large": ["facebook/bart-large", BartConfig, BartTokenizer, BartForConditionalGeneration],
}


class EncDecModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        
        print("Loading the encoder & the tokenizer...")
        self.args = args
        seed_everything(self.args.seed, workers=True)
        
        model_map = model_maps[self.args.model_name]
        config = model_map[1].from_pretrained(model_map[0])
        self.tokenizer = model_map[2].from_pretrained(model_map[0])
        self.model = model_map[3].from_pretrained(model_map[0])
        
        self.args.hidden_size = config.d_model
        self.args.max_encoder_len = min(self.args.max_encoder_len, config.max_position_embeddings)
        self.args.max_decoder_len = min(self.args.max_decoder_len, config.max_position_embeddings)
        
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.bos_token = self.tokenizer.bos_token
        self.args.eos_token = self.tokenizer.eos_token
        self.args.pad_token = self.tokenizer.pad_token
        self.args.unk_token = self.tokenizer.unk_token
        self.args.sep_token = self.tokenizer.sep_token
        self.args.bos_id = vocab[self.args.bos_token]
        self.args.eos_id = vocab[self.args.eos_token]
        self.args.pad_id = vocab[self.args.pad_token]
        self.args.unk_id = vocab[self.args.unk_token]
        self.args.sep_id = vocab[self.args.sep_token]
        
        self.class_linear = nn.Linear(self.args.hidden_size, self.args.num_classes)
        nn.init.xavier_uniform_(self.class_linear.weight)
        
        self.class_loss = nn.CrossEntropyLoss()
        
        self.test_samples = []
        
        self.save_hyperparameters(args)
        
    def forward(self, src_ids):  # src_ids: (B, S_L)
        src_masks = (src_ids != self.args.pad_id)  # (B, S_L)
        outputs = self.model.generate(
            input_ids=src_ids, attention_mask=src_masks,
            num_beams=self.args.beam_size, max_length=self.args.max_decoder_len, early_stopping=True,
            output_hidden_states=True, output_scores=True, return_dict_in_generate=True,
        )  # (B, T_L)
        
        lm_outputs, encoder_outputs = outputs.sequences, outputs.encoder_hidden_states[-1]  # (B, T_L), (B, S_L, d_h)
        class_logits = self.class_linear(encoder_outputs[:, 0])  # (B, C)
        class_preds = torch.max(class_logits, dim=-1).indices  # (B)
        
        return lm_outputs, class_preds
    
    def make_tokens(self, ids_list):
        tokens_list = []
        for ids in ids_list:
            token_ids = [id for id in ids if id >=0 and id < self.args.vocab_size]
            utter = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            tokens_list.append(self.tokenizer.tokenize(utter))
            
        return tokens_list

    def training_step(self, batch, batch_idx):
        src_ids, trg_ids, class_labels = batch  # (B, S_L), (B, T_L), (B)
        src_masks = (src_ids != self.args.pad_id)  # (B, S_L)
        outputs = self.model(input_ids=src_ids, attention_mask=src_masks, labels=trg_ids, output_hidden_states=True)
        lm_loss = outputs.loss
        logits = outputs.logits  # (B, T_L, d_h)
        encoder_outputs = outputs.encoder_last_hidden_state  # (B, S_L, d_h)
        
        class_logits = self.class_linear(encoder_outputs[:, 0])  # (B, C)
        
        batch_size = trg_ids.shape[0]
        class_loss = self.class_loss(class_logits, class_labels)
        loss = lm_loss + self.args.mtl_factor * class_loss
        
        lm_preds = torch.max(logits, dim=-1).indices  # (B, T_L)
        class_preds = torch.max(class_logits, dim=-1).indices  # (B)
        
        return {
            'loss': loss, 'lm_loss': lm_loss.detach(), 'class_loss': class_loss.detach(),
            'lm_preds': lm_preds, 'lm_trues': trg_ids[:, 1:],
            'class_preds': class_preds, 'class_trues': class_labels,
        }
    
    def training_epoch_end(self, training_step_outputs):
        train_losses, train_lm_losses, train_class_losses = [], [], []
        train_lm_preds, train_lm_trues = [], []
        train_class_preds, train_class_trues = [], []
        
        for result in training_step_outputs:
            train_losses.append(result['loss'].item())
            train_lm_losses.append(result['lm_loss'].item())
            train_class_losses.append(result['class_loss'].item())
            train_lm_preds += result['lm_preds'].tolist()
            train_lm_trues += result['lm_trues'].tolist()
            train_class_preds += result['class_preds'].tolist()
            train_class_trues += result['class_trues'].tolist()
        
        self.log('train_loss', np.mean(train_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_lm_loss', np.mean(train_lm_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_class_loss', np.mean(train_class_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        train_lm_preds = self.make_tokens(train_lm_preds)
        train_lm_trues = self.make_tokens(train_lm_trues)
        
        train_acc = get_accuracy(train_class_preds, train_class_trues)
        train_bleu = get_bleu(train_lm_preds, train_lm_trues)
        
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_bleu', train_bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def validation_step(self, batch, batch_idx):
        src_ids, trg_ids, class_labels = batch  # (B, S_L), (B, T_L), (B)
        lm_outputs, class_preds = self.forward(src_ids)

        return {
            'lm_preds': lm_outputs, 'lm_trues': trg_ids,
            'class_preds': class_preds, 'class_trues': class_labels,
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        valid_lm_preds, valid_lm_trues = [], []
        valid_class_preds, valid_class_trues = [], []
        
        for result in validation_step_outputs:
            valid_lm_preds += result['lm_preds'].tolist()
            valid_lm_trues += result['lm_trues'].tolist()
            valid_class_preds += result['class_preds'].tolist()
            valid_class_trues += result['class_trues'].tolist()
            
        valid_lm_preds = self.make_tokens(valid_lm_preds)
        valid_lm_trues = self.make_tokens(valid_lm_trues)
        
        valid_acc = get_accuracy(valid_class_preds, valid_class_trues)
        valid_bleu = get_bleu(valid_lm_preds, valid_lm_trues)
        
        self.log('valid_acc', valid_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_bleu', valid_bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def test_step(self, batch, batch_idx):
        src_ids, trg_ids, class_labels = batch  # (B, S_L), (B, T_L), (B), (B)
        lm_outputs, class_preds = self.forward(src_ids)

        return {
            'lm_preds': lm_outputs, 'lm_trues': trg_ids,
            'class_preds': class_preds, 'class_trues': class_labels,
        }
    
    def test_epoch_end(self, test_step_outputs):
        test_lm_preds, test_lm_trues = [], []
        test_class_preds, test_class_trues = [], []
        
        for result in test_step_outputs:
            test_lm_preds += result['lm_preds'].tolist()
            test_lm_trues += result['lm_trues'].tolist()
            test_class_preds += result['class_preds'].tolist()
            test_class_trues += result['class_trues'].tolist()
            
        test_lm_preds = self.make_tokens(test_lm_preds)
        test_lm_trues = self.make_tokens(test_lm_trues)
        
        test_acc = get_accuracy(test_class_preds, test_class_trues)
        test_bleu = get_bleu(test_lm_preds, test_lm_trues)
        
        self.log('test_acc', test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_bleu', test_bleu, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        zipped = list(zip(test_class_preds, test_class_trues, test_lm_preds, test_lm_trues))
        test_samples = random.sample(zipped, self.args.num_samples)
        for sample in test_samples:
            self.test_samples.append({
                "Predicted class": sample[0],
                "Actual class": sample[1],
                "Corrected": self.tokenizer.convert_tokens_to_string(sample[2]),
                "Actual": self.tokenizer.convert_tokens_to_string(sample[3]),
            })
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        if self.args.warmup_steps < 0.0:
            return [optimizer]
        else:
            scheduler = {
                'scheduler': get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=self.args.total_train_steps
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1

            }

            return [optimizer], [scheduler]