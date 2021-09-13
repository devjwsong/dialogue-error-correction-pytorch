from enc_module import *
from encdec_module import *

import torch
import argparse
import os
import natsort


def extract(args):
    print(f"Extracting the checkpont info from log {args.log_idx}...")
    ckpt_dir = f"lightning_logs/version_{args.log_idx}/checkpoints"
    if args.ckpt_name is None:
        ckpt_list = [ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")]
        ckpt_list = natsort.natsorted(ckpt_list)
        args.ckpt_name = ckpt_list[-1][:-5]
    
    model_name = args.ckpt_name.split('_')[0]
    if 'bert' in model_name:
        module = EncModule.load_from_checkpoint(f"{ckpt_dir}/{args.ckpt_name}.ckpt")
    elif 'bart' in model_name:
        module = EncDecModule.load_from_checkpoint(f"{ckpt_dir}/{args.ckpt_name}.ckpt")
        
    model = module.model
    tokenizer = module.tokenizer
    
    print("CHECK!")
    print(f"Model class: {type(model)}")
    print(f"Tokenizer class: {type(tokenizer)}")
    
    output_dir = f"lightning_logs/version_{args.log_idx}/{args.ckpt_name}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log_idx', type=int, required=True)
    parser.add_argument('--ckpt_name', type=str, required=False)
    
    args = parser.parse_args()
    
    extract(args)
    
    print("FINISHED.")