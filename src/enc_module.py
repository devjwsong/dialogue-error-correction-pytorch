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
import pickle


# 0: path, 1: config, 2: tokenizer, 3: model
model_maps = {
    "bert": ["bert-base-uncased", BertConfig, BertTokenizer, BertModel],
    "todbert": ["TODBERT/TOD-BERT-JNT-V1", AutoConfig, AutoTokenizer, AutoModel],
    "convbert": ["convbert", BertConfig, BertTokenizer, BertModel],
}


class EncModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
        
        print("Loading the encoder & the tokenizer...")
        self.args = args
        model_map = model_maps[self.args.model_name]
        config = model_map[1].from_pretrained(model_map[0])
        self.tokenizer = model_map[2].from_pretrained(model_map[0])
        self.encoder = model_map[3].from_pretrained(model_map[0])
        
        self.args.hidden_size = config.hidden_size
        self.args.max_encoder_len = min(self.args.max_encoder_len, config.max_position_embeddings)
        
        vocab = self.tokenizer.get_vocab()
        self.args.vocab_size = len(vocab)
        self.args.cls_token = self.tokenizer.cls_token
        self.args.sep_token = self.tokenizer.sep_token
        self.args.pad_token = self.tokenizer.pad_token
        self.args.unk_token = self.tokenizer.unk_token
        self.args.cls_id = vocab[self.args.cls_token]
        self.args.sep_id = vocab[self.args.sep_token]
        self.args.pad_id = vocab[self.args.pad_token]
        self.args.unk_id = vocab[self.args.unk_token]
        
        print("Loading the decoder...")
        seed_everything(self.args.seed, workers=True)
        self.decoder = nn.GRU(
            input_size=self.args.hidden_size,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_decoder_layers,
            dropout=self.args.decoder_dropout if self.args.num_decoder_layers > 1 else 0.0
        )
        self.attention = Attention(self.args.hidden_size)
        if self.args.use_copy:
            self.copy_mech = CopyMechanism(self.args.hidden_size)
        
        self.decoder_linear = nn.Linear(self.args.hidden_size * 2, self.args.vocab_size)
        self.encoder_linear = nn.Linear(self.args.hidden_size, self.args.num_classes)
        for name, param in self.decoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_uniform_(self.decoder_linear.weight)
        nn.init.xavier_uniform_(self.encoder_linear.weight)
        
        self.lm_loss = nn.NLLLoss(ignore_index=self.args.pad_id, reduction=self.args.loss_reduction)
        self.class_loss = nn.CrossEntropyLoss()
        
        self.test_samples = []
        
        self.save_hyperparameters(args)
    
    def forward(self, src_ids):  # src_ids: (B, S_L)
        encoder_outputs = self.forward_encoder(src_ids)  # (B, S_L, d_h)
        class_logits = self.encoder_linear(encoder_outputs[:, 0])  # (B, C)
        class_preds = torch.max(class_logits, dim=-1).indices  # (B)
        
        lm_outputs = self.forward_decoder(src_ids, encoder_outputs)  # (B, T_L)
        
        return lm_outputs, class_preds
    
    def forward_encoder(self, src_ids):  # src_ids: (B, S_L)
        encoder_masks = (src_ids != self.args.pad_id)  # (B, S_L)
        hidden_states = self.encoder(input_ids=src_ids, attention_mask=encoder_masks)[0]  # (B, S_L, d_h)
        
        return hidden_states  # (B, S_L, d_h)
    
    def step_decoder(self, trg_ids, src_ids, encoder_outputs):  # (B, T_L), (B, S_L), (B, S_L, d_h)
        batch_size, trg_len = trg_ids.shape[0], trg_ids.shape[1]
        decoder_outputs = torch.zeros(trg_len, batch_size, self.args.vocab_size).to(trg_ids.device)  # (T_L, B, V)
        
        input_ids = trg_ids[:, 0]  # (B)
        hidden = torch.zeros(self.args.num_decoder_layers, batch_size, self.args.hidden_size).to(trg_ids.device)  # (L, B, d_h)
        for t in range(1, trg_len):
            input_embs = self.encoder.embeddings.word_embeddings(input_ids)  # (B, d_h)
            
            outputs, hidden = self.decoder(input_embs.unsqueeze(0), hidden)  # (1, B, d_h), (L, B, d_h)
            context_vecs, attn_dists = self.attention(hidden[-1,:,:], encoder_outputs)  # (B, d_h), (B, S_L)
            
            vocab_dists = F.softmax(self.decoder_linear(torch.cat((hidden[-1,:,:], context_vecs), dim=-1)), dim=-1)  # (B, V)
            if self.args.use_copy:
                vocab_dists = self.copy_mech(context_vecs, hidden[-1,:,:], input_embs, vocab_dists, attn_dists, src_ids, self.args.pad_id)  # (B, V)
            
            decoder_outputs[t-1] = vocab_dists
            input_ids = trg_ids[:, t]  # (B)
                
        return decoder_outputs.transpose(0, 1)  # (B, T_L, V)
    
    def forward_decoder(self, src_ids, encoder_outputs): # (B, S_L), (B, S_L, d_h)
        decoder_results = []
        batch_size = src_ids.shape[0]
        
        for b in range(batch_size):
            result = self.beam_search(src_ids[b], encoder_outputs[b])
            decoder_results.append(result)
            
        decoder_results = torch.nn.utils.rnn.pad_sequence(decoder_results, batch_first=True, padding_value=self.args.pad_id)
        
        return decoder_results  # (B, T_L)
    
    def beam_search(self, src_ids, encoder_outputs):  # (S_L)  (S_L, d_h)
        src_ids = src_ids.repeat(self.args.beam_size, 1)  # (K, S_L)
        encoder_outputs = encoder_outputs.repeat(self.args.beam_size, 1, 1)  # (K, S_L, d_h)
        
        result_queue = PriorityQueue()
        cur_queue = PriorityQueue()
        for k in range(self.args.beam_size):
            cur_queue.put(BeamNode(self.args.cls_id, -0.0, [self.args.cls_id], self.args.num_decoder_layers, self.args.hidden_size))
        
        for t in range(1, self.args.max_decoder_len):
            input_ids = torch.full((self.args.beam_size, ), self.args.pad_id).to(src_ids.device)  # (K)
            hidden = torch.zeros(self.args.beam_size, self.args.num_decoder_layers, self.args.hidden_size).to(src_ids.device)  # (K, L, d_h)
            nodes = []
            for k in range(self.args.beam_size):
                node = cur_queue.get()
                nodes.append(node)
                input_ids[k] = node.decoded[-1]
                hidden[k] = node.hidden
            hidden = hidden.transpose(0, 1).contiguous()  # (L, K, d_h)
                
            input_embs = self.encoder.embeddings.word_embeddings(input_ids)  # (K, d_h)
            outputs, hidden = self.decoder(input_embs.unsqueeze(0), hidden)  # (1, K, d_h), (L, K, d_h)
            
            context_vecs, attn_dists = self.attention(hidden[-1,:,:], encoder_outputs)  # (K, d_h), (K, S_L)
            
            vocab_dists = F.softmax(self.decoder_linear(torch.cat((hidden[-1,:,:], context_vecs), dim=-1)), dim=-1)  # (K, V)
            if self.args.use_copy:
                vocab_dists = self.copy_mech(context_vecs, hidden[-1,:,:], input_embs, vocab_dists, attn_dists, src_ids, self.args.pad_id)  # (K, V)
            
            vocab_dists = torch.log(vocab_dists)  # (K, V)
            next_log_probs, next_ids = torch.topk(vocab_dists, self.args.beam_size, dim=-1)  # (K, K), (K, K)
            next_log_probs, next_ids = next_log_probs.view(-1), next_ids.view(-1)  # (K * K), (K * K)
            
            cur_queue = PriorityQueue()
            for k in range(self.args.beam_size * self.args.beam_size):
                node_idx = k // self.args.beam_size
                token_id = next_ids[k].item()
                new_node = BeamNode(
                    token_id, -(-nodes[node_idx].prob + next_log_probs[k].item()), nodes[node_idx].decoded + [token_id], 
                    self.args.num_decoder_layers, self.args.hidden_size
                )
                new_node.hidden = hidden.transpose(0, 1)[node_idx]
                if token_id == self.args.sep_id:
                    new_node.prob = new_node.prob / float(len(new_node.decoded))
                    result_queue.put(new_node)
                else:
                    cur_queue.put(new_node)

            if len(result_queue) == self.args.beam_size:
                break
                
        if len(result_queue) < self.args.beam_size:
            left = self.args.beam_size - len(result_queue)
            for i in range(left):
                node = cur_queue.get()
                node.prob = node.prob / float(len(node.decoded))
                result_queue.put(node)
        
        result = result_queue.get().decoded
        if result[0] == self.args.cls_id:
            result = result[1:]
        result = torch.LongTensor(result)
        
        return result
    
    def make_tokens(self, ids_list):
        token_list = []
        for ids in ids_list:
            tokens = self.tokenizer.convert_ids_to_tokens(ids)
            new_tokens = []
            for token in tokens:
                if token == self.args.sep_token or token == self.args.pad_token:
                    break

                new_tokens.append(token)

            token_list.append(new_tokens)

        return token_list

    def training_step(self, batch, batch_idx):
        src_ids, trg_ids, class_labels = batch  # (B, S_L), (B, T_L), (B)
        encoder_outputs = self.forward_encoder(src_ids)  # (B, S_L, d_h)
        lm_outputs = self.step_decoder(trg_ids, src_ids, encoder_outputs)  # (B, T_L, V)
        class_logits = self.encoder_linear(encoder_outputs[:, 0])  # (B, C)
        
        if self.current_epoch == 7:
            with open(f"{self.logger.log_dir}/encoder_outputs.pickle", 'wb') as f:
                pickle.dump(encoder_outputs, f)
            with open(f"{self.logger.log_dir}/lm_outputs.pickle", 'wb') as f:
                pickle.dump(encoder_outputs, f)
            with open(f"{self.logger.log_dir}/class_logits.pickle", 'wb') as f:
                pickle.dump(encoder_outputs, f)
            
        
        batch_size = trg_ids.shape[0]
        lm_loss = self.lm_loss(torch.log(lm_outputs[:, :-1]).contiguous().view(-1, self.args.vocab_size), trg_ids[:, 1:].contiguous().view(-1))  # (B * T_L)
        class_loss = self.class_loss(class_logits, class_labels)
        loss = lm_loss + self.args.mtl_factor * class_loss
        
        lm_preds = torch.max(lm_outputs[:, :-1], dim=-1).indices  # (B, T_L-1)
        class_preds = torch.max(class_logits, dim=-1).indices  # (B)
        
        return {
            'loss': loss, 'lm_loss': lm_loss, 'class_loss': class_loss,
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
            'lm_preds': lm_outputs[:, :-1], 'lm_trues': trg_ids[:, 1:],
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
            'lm_preds': lm_outputs[:, :-1], 'lm_trues': trg_ids[:, 1:],
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
        
        
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.decoder_proj = nn.Linear(hidden_size, hidden_size)
        self.encoder_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.decoder_proj.weight)
        nn.init.xavier_uniform_(self.encoder_proj.weight)

    def forward(self, decoder_hidden, encoder_outputs, encoder_masks=None):  # (B, d_h), (B, S_L, d_h), (B, S_L)
        query = self.decoder_proj(decoder_hidden)  # (B, d_h)
        key = self.encoder_proj(encoder_outputs)  # (B, S_L, d_h)

        energy = torch.sum(torch.mul(key, query.unsqueeze(1)), dim=-1)  # (B, S_L)
        if encoder_masks is not None:
            energy.masked_fill_(~encoder_masks, -1*1e7)

        attn_dists = F.softmax(energy, dim=-1)  # (B, S_L)
        context_vecs = torch.sum(torch.mul(encoder_outputs, attn_dists.unsqueeze(2)), dim=1)  # (B, d_h)

        return context_vecs, attn_dists
    
    
class CopyMechanism(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w_h = nn.Linear(hidden_size, 1, bias=False)
        self.w_s = nn.Linear(hidden_size, 1, bias=False)
        self.w_x = nn.Linear(hidden_size, 1, bias=True)
        nn.init.xavier_uniform_(self.w_h.weight)
        nn.init.xavier_uniform_(self.w_s.weight)
        nn.init.xavier_uniform_(self.w_x.weight)

    def forward(self, context_vecs, hidden, trg_embs, vocab_dists, attn_dists, src_ids, pad_id):  # (B, d_h), (B, d_h), (B, d_h), (B, V), (B, S_L), (B, S_L)
        context_feats = self.w_h(context_vecs)  # (B, 1)
        decoder_feats = self.w_s(hidden)  # (B, 1)
        trg_feats = self.w_x(trg_embs)  # (B, 1)
        gen_feats = context_feats + decoder_feats + trg_feats  # (B, 1)
        p_gen = torch.sigmoid(gen_feats)  # (B, 1)
        
        vocab_dists = p_gen * vocab_dists  # (B, V)
        attn_dists = (1 - p_gen) * attn_dists  # (B, S_L)
        
        final_dists = vocab_dists.scatter_add_(dim=-1, index=src_ids, src=attn_dists)  # (B, V)
        
        return final_dists