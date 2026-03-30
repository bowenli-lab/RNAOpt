import warnings
warnings.filterwarnings("ignore")

from model import RNA_MaskedLM_pretrain
from utils import RNADataset_pretrain, RNATokenizer, load_config

from torch.utils.data import DataLoader
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to the config file"
    )

    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Configuration loaded successfully from: {args.config}")
    else:
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    print("Configuration loaded successfully.")
    

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config['seed'])
    tokenizer = RNATokenizer()

    train_dataset = RNADataset_pretrain(
        hdf5_path=config['train_file'],
        tokenizer=tokenizer,
        max_len=config['max_len'],
        mask_prob=config['mask_prob']
    )

    train_loader = DataLoader(train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True)

    val_dataset = RNADataset_pretrain(
        hdf5_path=config['val_file'],
        tokenizer=tokenizer,
        max_len=config['max_len'],
        mask_prob=config['mask_prob']
    )
    val_loader = DataLoader(val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True)

    test_dataset = RNADataset_pretrain(
        hdf5_path=config['test_file'],
        tokenizer=tokenizer,
        max_len=config['max_len'],
        mask_prob=config['mask_prob']
    )
    test_loader = DataLoader(test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True)

    model = RNA_MaskedLM_pretrain(config=config)

    checkpoint_callback = ModelCheckpoint(
        monitor=config['monitor'],
        dirpath=config['ckpt_dir'],
        filename=config['ckpt_filename'],
        save_top_k=config['save_top_k'],
        mode=config['mode'],
        every_n_train_steps=config['every_n_steps']
    )

    trainer = pl.Trainer(
        max_steps=config['total_steps'], 
        default_root_dir=config['log_dir'], 
        accelerator=config['accelerator'], 
        devices=config['devices'], 
        log_every_n_steps=config['log_every_n_steps'],
        strategy=config['strategy'], #'auto',
        accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=[checkpoint_callback],
        val_check_interval=config['val_check_interval'], 
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)
    
    