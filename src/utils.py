from sklearn.metrics import f1_score
from torchtext.data.metrics import bleu_score

import heapq
import torch


def get_f1(preds, trues):
    assert len(preds) == len(trues)
    
    return f1_score(trues, preds)


def get_bleu(preds, trues):
    assert len(preds) == len(trues)
    
    refers = [[true] for true in trues]
    return bleu_score(preds, refers)


class BeamNode():
    def __init__(self, cur_idx, prob, decoded, num_decoder_layers, hidden_size):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.hidden = torch.zeros(num_decoder_layers, hidden_size)
        
    def __gt__(self, other):
        return self.prob > other.prob
    
    def __ge__(self, other):
        return self.prob >= other.prob
    
    def __lt__(self, other):
        return self.prob < other.prob
    
    def __le__(self, other):
        return self.prob <= other.prob
    
    def __eq__(self, other):
        return self.prob == other.prob
    
    def __ne__(self, other):
        return self.prob != other.prob
    
    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")
    

class PriorityQueue():
    def __init__(self):
        self.queue = []
        
    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))
        
    def get(self):
        return heapq.heappop(self.queue)[1]
    
    def __len__(self):
        return len(self.queue)
    
    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)
        
    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)
        