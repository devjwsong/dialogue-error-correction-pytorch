from torch.utils.data import DataLoader
from enc_module import *
from encdec_module import *
from custom_dataset import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

import argparse
import json
import os
import pickle


def run(args):
    # For model setting
    with open(f"{args.data_dir}/class_dict.json", 'r') as f:
        class_dict = json.load(f)
    args.num_classes = len(class_dict)
    
    print(f"Loading training module with the encoder {args.model_name}...")
    if 'bert' in args.model_name:
        module = EncModule(args)
        ppd = EncPadCollate(pad_id=args.pad_id)
    elif 'bart' in args.model_name:
        module = EncDecModule(args)
        ppd = EncDecPadCollate(pad_id=args.pad_id)
    
    print("Loading datasets...")
    # For data loading    
    train_set = CustomDataset(args, "train", module.tokenizer)
    valid_set = CustomDataset(args, "valid", module.tokenizer)
    test_set = CustomDataset(args, "test", module.tokenizer)

    # Dataloaders
    seed_everything(args.seed, workers=True)
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, collate_fn=ppd.pad_collate, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    # Calculate total training steps
    args.gpus = [int(idx) for idx in args.gpus]
    num_gpus = len(args.gpus)
    num_devices = num_gpus * args.num_nodes
    q, r = divmod(len(train_loader), num_devices)
    num_batches = q if r == 0 else q+1
    args.total_train_steps = args.num_epochs * num_batches
    args.warmup_steps = int(args.warmup_prop * args.total_train_steps)

    print("Setting pytorch lightning callback & trainer...")
    # Model checkpoint callback
    filename = "best_ckpt_{epoch}_{valid_bleu:.4f}_{valid_acc:.4f}"
    monitor = "valid_bleu"
    
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode='max',
        every_n_epochs=1,
        save_weights_only=True
    )
    
    stopping_callback = EarlyStopping(
        monitor=monitor,
        min_delta=1e-4,
        patience=3,
        verbose=True,
        mode='max'
    )
    
    # Trainer setting
    trainer = Trainer(
        check_val_every_n_epoch=1,
        gpus=args.gpus,
        auto_select_gpus=True,
        num_nodes=args.num_nodes,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.max_grad_norm,
        num_sanity_val_steps=0,
        deterministic=True,
        accelerator="ddp",
        callbacks=[checkpoint_callback, stopping_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    
    print("Train starts.")
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print("Training done.")
    
    print("Test starts.")
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    print("Saving the test samples...")
    file_path = f"{module.logger.log_dir}/test_samples.pickle"
    if not os.path.isfile(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(module.test_samples, f)
    
    print("GOOD BYE.")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--data_dir', type=str, default="data", help="The directory path to save pickle files.")
    parser.add_argument('--model_name', required=True, type=str, help="The model to test.")
    parser.add_argument('--num_epochs', type=int, default=5, help="The number of total epochs.")
    parser.add_argument('--train_batch_size', type=int, default=16, help="The batch size for training.")
    parser.add_argument('--eval_batch_size', type=int, default=4, help="The batch size for inferencing.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--max_encoder_len', type=int, default=512, help="The maximum length of a source sequence.")
    parser.add_argument('--num_decoder_layers', type=int, default=1, help="The number of layers for the GRU decoder.")
    parser.add_argument('--decoder_dropout', type=float, default=0.0, help="The dropout rate for the GRU decoder.")
    parser.add_argument('--max_decoder_len', type=int, default=256, help="The maximum length of a target sequence.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="The starting learning rate.")
    parser.add_argument('--warmup_prop', type=float, default=0.0, help="The warmup step proportion.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--use_copy', action='store_true', help="Using copy or not?")
    parser.add_argument('--mtl_factor', type=float, default=1.0, help="The loss factor for multi-task learning.")
    parser.add_argument('--loss_reduction', type=str, default='mean', help="How to reduce the LM loss value?")
    parser.add_argument('--beam_size', type=int, default=4, help="The beam size for the beam search when inferencing.")
    parser.add_argument('--num_samples', type=int, default=100, help="The number of test samples to show.")
    parser.add_argument('--gpus', nargs="+", default=["0"], help="The indices of GPUs to use.")
    parser.add_argument('--num_nodes', type=int, default=1, help="The number of machine.")
    
    args = parser.parse_args()
    
    assert args.model_name in ["bert", "todbert", "convbert", "bart-base", "bart-large"]
    if "bert" in args.model_name:
        assert args.num_decoder_layers is not None and args.decoder_dropout is not None
    assert args.loss_reduction in ["mean", "sum"]
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
    run(args)