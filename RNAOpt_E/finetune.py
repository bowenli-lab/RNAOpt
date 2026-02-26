from model import RNA_MaskedLM_finetune
from utils import RNADataset_finetune, RNATokenizer, load_config

from torch.utils.data import DataLoader
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    config = load_config("/home/reagan/Projects/RNA_optimization/model/models_put_on_github/model_modules/configs/finetune_config.yaml")
    print("Configuration loaded successfully.")

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(config['seed'])
    tokenizer = RNATokenizer()

    train_dataset = RNADataset_finetune(
        csv_file=config['train_file'],
        tokenizer=tokenizer,
        max_len=config['max_len']
    )

    train_loader = DataLoader(train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True)

    val_dataset = RNADataset_finetune(
        csv_file=config['val_file'],
        tokenizer=tokenizer,
        max_len=config['max_len']
    )
    val_loader = DataLoader(val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True)

    test_dataset = RNADataset_finetune(
        csv_file=config['test_file'],
        tokenizer=tokenizer,
        max_len=config['max_len']
    )
    test_loader = DataLoader(test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True)

    model = RNA_MaskedLM_finetune(config=config)
    checkpoint = torch.load(config['pretrained_model_path'])
    full_state_dict = checkpoint['state_dict']

    missing_keys = model.load_state_dict(full_state_dict, strict=False).missing_keys
    print(f"Ignoring missing keys: {missing_keys}")
    

    checkpoint_callback = ModelCheckpoint(
        monitor=config['monitor'],
        dirpath=config['ckpt_dir'],
        filename=config['ckpt_filename'],
        save_top_k=config['save_top_k'],
        mode=config['mode'],
        every_n_train_steps=config['every_n_steps']
    )

    trainer = pl.Trainer(
        max_epochs=config['max_epochs'], 
        default_root_dir=config['log_dir'], 
        accelerator=config['accelerator'], 
        devices=config['devices'],
        strategy=config['strategy'], #'auto',
        accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=[checkpoint_callback],
        val_check_interval=config['val_check_interval'], 
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)