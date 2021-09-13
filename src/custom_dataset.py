from torch.utils.data import Dataset
from tqdm import tqdm

import torch
import pickle


class CustomDataset(Dataset):
    def __init__(self, args, data_type, tokenizer):
        if "bert" in args.model_name:
            bos_token = args.cls_token
            eos_token = args.sep_token
            sep_token = args.sep_token
        elif "bart" in args.model_name:
            bos_token = args.bos_token
            eos_token = args.eos_token
            sep_token = args.sep_token
        
        with open(f"{args.data_dir}/{data_type}.pickle", 'rb') as f:
            data = pickle.load(f)
            
        self.src_ids = []  # (N, S_L)
        self.trg_ids = []  # (N, T_L)
        self.class_labels = []  # (N)
        
        for class_label, obj in data.items():
            print(f"Class: {class_label}")
            
            src_states = obj["src_states"]
            src_utters = obj["src_utters"]
            trg_utters = obj["trg_utters"]
            
            for i in tqdm(range(len(src_states))):
                src_state = src_states[i]
                src_utter = src_utters[i]
                trg_utter = trg_utters[i]
                
                self.class_labels.append(class_label)
                states = []
                for pair in src_state:
                    states.append(f"{pair[0]}: {pair[1]}")
                state_seq = " ".join(states)
                state_tokens = tokenizer.tokenize(state_seq)
                utter_tokens = tokenizer.tokenize(src_utter)
                src_tokens = [bos_token] + state_tokens + [sep_token] + utter_tokens + [eos_token]
                src_ids = tokenizer.convert_tokens_to_ids(src_tokens)
                
                trg_tokens = [bos_token] + tokenizer.tokenize(trg_utter) + [eos_token]
                trg_ids = tokenizer.convert_tokens_to_ids(trg_tokens)
                
                if len(src_ids) > args.max_encoder_len:
                    src_ids = src_ids[args.max_encoder_len]
                    src_ids[-1] = tokenizer.get_vocab()[eos_token]
                    
                if len(trg_ids) > args.max_decoder_len:
                    trg_ids = trg_ids[args.max_decoder_len]
                    trg_ids[-1] = tokenizer.get_vocab()[eos_token]
                
                self.src_ids.append(src_ids)
                self.trg_ids.append(trg_ids)
                
        assert len(self.src_ids) == len(self.trg_ids)
        assert len(self.src_ids) == len(self.class_labels)
        
    def __len__(self):
        return len(self.src_ids)
    
    def __getitem__(self, i):
        return self.src_ids[i], self.trg_ids[i], self.class_labels[i]
    
    
class EncPadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
        # Padding
        src_ids, trg_ids, class_labels = [], [], []
        for idx, tup in enumerate(batch):
            src_ids.append(torch.LongTensor(tup[0]))
            trg_ids.append(torch.LongTensor(tup[1]))
            class_labels.append(tup[2])

        src_ids = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=self.pad_id)
        trg_ids = torch.nn.utils.rnn.pad_sequence(trg_ids, batch_first=True, padding_value=self.pad_id)
        class_labels = torch.LongTensor(class_labels)

        # set contiguous for memory efficiency
        return src_ids.contiguous(), trg_ids.contiguous(), class_labels.contiguous()
    
    
class EncDecPadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
        # Padding
        src_ids, trg_ids, class_labels = [], [], []
        for idx, tup in enumerate(batch):
            src_ids.append(torch.LongTensor(tup[0]))
            trg_ids.append(torch.LongTensor(tup[1]))
            class_labels.append(tup[2])

        src_ids = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=self.pad_id)
        trg_ids = torch.nn.utils.rnn.pad_sequence(trg_ids, batch_first=True, padding_value=-100)
        class_labels = torch.LongTensor(class_labels)

        # set contiguous for memory efficiency
        return src_ids.contiguous(), trg_ids.contiguous(), class_labels.contiguous()
