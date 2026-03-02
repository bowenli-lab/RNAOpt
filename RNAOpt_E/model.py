import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import JambaModel

import lightning as pl
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR




class BiJamba(nn.Module):
    def __init__(self, config):
        super(BiJamba, self).__init__()
        # self.all_config = config
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.d_model = self.config.hidden_size
        
        self.jamba = JambaModel(self.config)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.linear_head = nn.Sequential(nn.Linear(self.d_model,128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.vocab_size)
                                            )

        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.d_model)
        )



    def forward(self, x, attention_mask):
        x = self.embedding(x)
    
        residual = x
        # Forward
        x_norm = self.norm1(x)
        jamba_out_forward = self.jamba(inputs_embeds = x_norm, attention_mask = attention_mask)['last_hidden_state']

        # Backward 
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        attention_mask_flip = torch.flip(attention_mask, dims=[1])
        
        jamba_out_backward = self.jamba(inputs_embeds = x_flip, attention_mask = attention_mask_flip)['last_hidden_state']

        jamba_out_backward = torch.flip(jamba_out_backward, dims=[1])  # Flip back

        # Combining forward and backward
        jamba_out = jamba_out_forward + jamba_out_backward
        
        jamba_out = self.norm2(jamba_out)
        ff_out = self.feed_forward(jamba_out)

        output0 = ff_out + residual
        output1 = self.linear_head(output0)
        return output1, output0


class BiJambaMLM(nn.Module):
    def __init__(self, config,):
        super().__init__()
        self.model = BiJamba(config)

    def forward(self, input_ids, device, attention_mask, check_error=False):

        if check_error:
            logits = self.model(seq)
            labels = None
            return logits, labels

        else:
            masked_seq = input_ids.to(device)
            
            logits, last_hidden_state = self.model(masked_seq, attention_mask) # shape are (batch_size/num_GPUs, max_len, d_model)
        
            return logits, last_hidden_state




class BiJambaForRegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BiJambaMLM(config)

        self.regression_head = nn.Sequential(nn.Linear(config.hidden_size,128),
                                            nn.GELU(),
                                            nn.Linear(128, 64),
                                            nn.GELU(),
                                            nn.Linear(64, 1)
                                            )
    
    def forward(self, input_ids, attention_mask):
        logits, last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask, device=input_ids.device)
        
        pooling = nn.AdaptiveAvgPool1d(1)
        pooled_output = pooling(last_hidden_state.transpose(1, 2))  # Shape: [batch, hidden_dim, 1]
        pooled_output = pooled_output.squeeze(-1) # Shape: [batch, hidden_dim]

        return self.regression_head(pooled_output), logits, last_hidden_state






class RNA_MaskedLM_finetune(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BiJambaForRegression(config=config['JambaConfig'])
        self.loss_fn = nn.MSELoss()  
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        value, logits, last_hidden_state = self.model(input_ids, attention_mask)
        return value, logits, last_hidden_state

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float()
        
        value, logits, _ = self(input_ids, attention_mask)

        loss = self.loss_fn(value.squeeze(), labels)  # Ignore -100
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss
        

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        value, logits, _ = self(input_ids, attention_mask)
        
        loss = self.loss_fn(value.squeeze(), labels)  # Ignore -100
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        value, logits, _ = self(input_ids, attention_mask)
        
        loss = self.loss_fn(value.squeeze(), labels)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['learning_rate']))
        return optimizer



class RNA_MaskedLM_pretrain(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.warmup_steps = float(config['warmup_steps'])
        self.total_steps = float(config['total_steps'])
        self.model = BiJambaMLM(config=config['JambaConfig'])
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()


    def forward(self, input_ids, attention_mask):
        logits, last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask, device=input_ids.device)
        return logits, last_hidden_state

    def training_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch

        logits, last_hidden_state = self(input_ids, attention_mask)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))  # Ignore -100
        
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, sync_dist=True)
             
        return loss

    
    def validation_step(self, batch, batch_idx):
        
        input_ids, attention_mask, labels = batch
    
        logits, last_hidden_state = self(input_ids, attention_mask)

        
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))  # Ignore -100
        self.log("val_loss", loss, sync_dist=True)
        
    
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['learning_rate']), weight_decay=float(self.config['weight_decay']))

        # Define function for warm-up
        lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(optimizer, start_factor=float(self.config['warmup_start_factor']), total_iters=float(self.warmup_steps)),
                    CosineAnnealingLR(optimizer, T_max=self.total_steps - self.warmup_steps),
                ],
                milestones=[self.warmup_steps],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


