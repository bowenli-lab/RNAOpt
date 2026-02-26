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
        #self.embedding = nn.Linear(self.all_config['max_len'], self.d_model)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        # self.linear_head = nn.Linear(self.d_model, self.all_config['vocab_size'])
        self.linear_head = nn.Sequential(nn.Linear(self.d_model,128),
                                            nn.ReLU(),
                                            nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, self.vocab_size)
                                            )
        #self.softmax = nn.Softmax(dim=-1)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.d_model)
        )



    def forward(self, x, attention_mask):
        x = self.embedding(x)  # (batch_size, max_len, d_model)
    
        residual = x
        # Forward
        x_norm = self.norm1(x)
        # print("Shape of x_norm:", x_norm.shape)
        jamba_out_forward = self.jamba(inputs_embeds = x_norm, attention_mask = attention_mask)['last_hidden_state']

        # Backward 
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        attention_mask_flip = torch.flip(attention_mask, dims=[1])
        
        jamba_out_backward = self.jamba(inputs_embeds = x_flip, attention_mask = attention_mask_flip)['last_hidden_state']

        jamba_out_backward = torch.flip(jamba_out_backward, dims=[1])  # Flip back

        # # Combining forward and backward
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

# class BiJambaForRegression_lora(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.model = BiJambaMLM(config)

#         self.regression_head = nn.Sequential(nn.Linear(config.hidden_size,128),
#                                             nn.GELU(),
#                                             nn.Linear(128, 64),
#                                             nn.GELU(),
#                                             nn.Linear(64, 1)
#                                             )
    
#     def forward(self, input_ids, attention_mask, **kwargs):
#         logits, last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask, device=input_ids.device)
        
#         pooling = nn.AdaptiveAvgPool1d(1)
#         pooled_output = pooling(last_hidden_state.transpose(1, 2))  # Shape: [batch, hidden_dim, 1]
#         pooled_output = pooled_output.squeeze(-1) # Shape: [batch, hidden_dim]

#         return self.regression_head(pooled_output)




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

        #breakpoint()
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
        self.loss_fn = nn.CrossEntropyLoss()  # Automatically ignores labels set to -100
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


# class RNA_MaskedLM_lora_finetune(pl.LightningModule):
#     def __init__(self, config, peft_config):
#         super().__init__()
#         self.config = config
#         self.before_lora_model = BiJambaForRegression_lora(config=config['JambaConfig'])
#         # for name, module in self.before_lora_model.named_modules():
#         #     print(name, "->", type(module))
#         checkpoint = torch.load(config['pretrained_checkpoint_path'])
#         full_state_dict = checkpoint['state_dict']

#         modified_state_dict = {}
#         for key, value in full_state_dict.items():
#             # Modify the key to remove the 'model.' prefix
#             new_key = key.replace('model.model.model', 'model.model')
#             modified_state_dict[new_key] = value
#         self.before_lora_model.load_state_dict(modified_state_dict, strict=False)

#         self.model = get_peft_model(self.before_lora_model, peft_config)
#         self.loss_fn = nn.MSELoss()  
#         self.save_hyperparameters()

#     def forward(self, input_ids, attention_mask):
#         value = self.model(input_ids, attention_mask)
#         return value

#     def training_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"].float()
        
#         value = self(input_ids, attention_mask)

#         #breakpoint()
#         loss = self.loss_fn(value.squeeze(), labels)  # Ignore -100
#         self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

#         return loss
        

#     def validation_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"]
        
#         value = self(input_ids, attention_mask)
        
#         loss = self.loss_fn(value.squeeze(), labels)  # Ignore -100
#         self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"]
        
#         value = self(input_ids, attention_mask)
        
#         loss = self.loss_fn(value.squeeze(), labels)
#         self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config['learning_rate']))
#         return optimizer

# class CLModel_pretrain(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = BiJamba(config['JambaConfig'])
#         self.temperature = config['temperature']
#         self.lr = float(config['learning_rate'])
#         print("Learning rate:", self.lr)
#         self.weight_decay = config['weight_decay']
#         self.warmup_steps = config['warmup_steps']
#         self.total_steps = config['total_steps']
#         self.warmup_start_factor = config['warmup_start_factor']
#         self.ckpt_dir = config['ckpt_dir']    
#         self.loss_fn = nn.CrossEntropyLoss()  # Automatically ignores labels set to -100
#         self.save_hyperparameters()

#     def forward(self, input_ids, attention_mask):
#         logits, last_hidden_state = self.encoder(x=input_ids, attention_mask=attention_mask)
#         pooling = nn.AdaptiveAvgPool1d(1)
#         pooled_output = pooling(last_hidden_state.transpose(1, 2))  # Shape: [batch, hidden_dim, 1]
#         pooled_output = pooled_output.squeeze(-1) # Shape: [batch, hidden_dim]

#         return logits, F.normalize(pooled_output, p=2, dim=1)

#     def training_step(self, batch, batch_idx):
#         logits_z1, z1 = self.forward(batch["input_ids_1"], batch["attention_mask_1"])
#         logits_z2, z2 = self.forward(batch["input_ids_2"], batch["attention_mask_2"])
#         # print("Check z1 and z2: ", torch.equal(z1, z2))
#         # print("Casual check: ", torch.allclose(z1, z2, atol=1e-4, rtol=1e-2))


#         batch_size = z1.size(0)
#         z = torch.cat([z1, z2], dim=0)  # shape: [2B, D]

#         # Compute cosine similarity
#         similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

#         # Mask self-similarity
#         mask = torch.eye(2 * batch_size, device=z.device).bool()
#         similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

#         # Build correct labels
#         labels = torch.arange(batch_size, device=z.device)
#         labels = torch.cat([labels + batch_size, labels], dim=0)  # size = [2B]

#         # Now similarity_matrix has shape [2B, 2B]
#         # and labels[i] gives the index of the positive sample for sample i

#         # Scale
#         similarity_matrix = similarity_matrix / self.temperature

#         # Compute loss
#         infoNCEloss = self.loss_fn(similarity_matrix, labels)
#         # print("Training step loss:", loss.item())

#         mlmloss_z1 = self.loss_fn(logits_z1.view(-1, logits_z1.size(-1)), batch['labels'].view(-1))  # Ignore -100
#         mlmloss_z2 = self.loss_fn(logits_z2.view(-1, logits_z2.size(-1)), batch['labels'].view(-1))  # Ignore -100


#         combined_loss = infoNCEloss + mlmloss_z1 + mlmloss_z2
#         # print("Training step loss:", combined_loss.item())

        
#         self.log("train_loss", combined_loss, on_step=True, on_epoch=False, sync_dist=True)
#         self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, sync_dist=True)

#         return combined_loss

#     # def on_train_end(self):
#         # current_step = self.global_step
#         # current_epoch = self.current_epoch

#         # # Get learning rate from the first param group
#         # current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

#         # You can log or print this info
#         # print(f"[Train End] Epoch: {current_epoch}, Step: {current_step}, LR: {current_lr:.6e}")

#         # If you also track training loss manually, log it here
#         # if "train_loss" in self.trainer.callback_metrics:
#         #     train_loss = self.trainer.callback_metrics["train_loss"]
#             # print(f"[Train End] Train Loss: {train_loss:.6f}")
        
#         # csv_path = os.path.join(self.log_dir, "metrics_log.csv")
#         # file_exists = os.path.isfile(csv_path)

#         # with open(csv_path, mode='a', newline='') as csvfile:
#         #     writer = csv.writer(csvfile)
#         #     # if not file_exists:
#         #     #     writer.writerow(["epoch", "step", "lr", "train_loss", "val_loss"])  # header
#         #     writer.writerow([current_epoch, current_step, f"{current_lr:.10e}", train_loss, ""])  # Assuming val_loss is not available at this point
        

#     def validation_step(self, batch, batch_idx):
#         logits_z1, z1 = self.forward(batch["input_ids_1"], batch["attention_mask_1"])
#         logits_z2, z2 = self.forward(batch["input_ids_2"], batch["attention_mask_2"])
#         # print("Check z1 and z2: ", torch.equal(z1, z2))
#         # print("Casual check: ", torch.allclose(z1, z2, atol=1e-4, rtol=1e-2))


#         batch_size = z1.size(0)
#         z = torch.cat([z1, z2], dim=0)  # shape: [2B, D]

#         # Compute cosine similarity
#         similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

#         # Mask self-similarity
#         mask = torch.eye(2 * batch_size, device=z.device).bool()
#         similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

#         # Build correct labels
#         labels = torch.arange(batch_size, device=z.device)
#         labels = torch.cat([labels + batch_size, labels], dim=0)  # size = [2B]

#         # Now similarity_matrix has shape [2B, 2B]
#         # and labels[i] gives the index of the positive sample for sample i

#         # Scale
#         similarity_matrix = similarity_matrix / self.temperature

#         # Compute loss
#         infoNCEloss = self.loss_fn(similarity_matrix, labels)
#         # print("Training step loss:", loss.item())

#         mlmloss_z1 = self.loss_fn(logits_z1.view(-1, logits_z1.size(-1)), batch['labels'].view(-1))  # Ignore -100
#         mlmloss_z2 = self.loss_fn(logits_z2.view(-1, logits_z2.size(-1)), batch['labels'].view(-1))  # Ignore -100


#         combined_loss = infoNCEloss + mlmloss_z1 + mlmloss_z2
#         # print("Validation step loss:", combined_loss.item())

#         # print("Validation step loss:", loss.item())
#         self.log("val_loss", combined_loss, on_step=False, on_epoch=True, sync_dist=True)
#         self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)

#         return combined_loss

#     def on_validation_end(self):
#         # Get metrics from trainer (safely)
#         val_loss = self.trainer.callback_metrics.get("val_loss", torch.tensor(float("nan")))
#         train_loss = self.trainer.callback_metrics.get("train_loss", torch.tensor(float("nan")))
#         # print(f"[Validation End] Train Loss: {train_loss}, Validation Loss: {val_loss}")
#         # Convert to float if tensor
#         val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else float(val_loss)
#         train_loss = train_loss.item() if isinstance(train_loss, torch.Tensor) else float(train_loss)

#         current_step = self.global_step
#         current_epoch = self.current_epoch

#         # Get learning rate from the first param group
#         # current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

#         # Build checkpoint filename
#         ckpt_name = f"epoch={self.current_epoch}-step={self.global_step}-train_loss={train_loss:.10f}-val_loss={val_loss:.10f}.ckpt"
#         ckpt_path = f"{self.ckpt_dir}/{ckpt_name}"

#         # Save checkpoint
#         self.trainer.save_checkpoint(ckpt_path)
        
#         # csv_path = os.path.join(self.log_dir, "metrics_log.csv")
#         # file_exists = os.path.isfile(csv_path)

#         # with open(csv_path, mode='a', newline='') as csvfile:
#         #     writer = csv.writer(csvfile)
#         #     if not file_exists:
#         #         writer.writerow(["epoch", "step", "lr", "train_loss", "val_loss"])  # header
#         #     writer.writerow([current_epoch, current_step, current_lr,train_loss, val_loss])
#         #     # print(f"[INFO] Checkpoint saved to: {ckpt_path}")


#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

#         # Define function for warm-up
#         lr_scheduler = SequentialLR(
#                 optimizer,
#                 schedulers=[
#                     LinearLR(optimizer, start_factor=self.warmup_start_factor, total_iters=self.warmup_steps),
#                     CosineAnnealingLR(optimizer, T_max=self.total_steps - self.warmup_steps),
#                 ],
#                 milestones=[self.warmup_steps],
#             )

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": lr_scheduler,
#                 "interval": "step",
#                 "frequency": 1,
#             },
#         }


# class CLModel_finetune(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = BiJambaForRegression(config['JambaConfig'])
#         self.lr = config['learning_rate']
#         self.weight_decay = config['weight_decay']
#         self.loss_fn = nn.MSELoss()  
#         self.config = config

#     def forward(self, input_ids, attention_mask):
#         value = self.model(input_ids, attention_mask)
#         return value

#     def training_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"].float()
        
#         value = self(input_ids, attention_mask)

#         #breakpoint()
#         loss = self.loss_fn(value.squeeze(), labels)  # Ignore -100
#         self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

#         return loss
        

#     def validation_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"]
        
#         value = self(input_ids, attention_mask)
        
#         loss = self.loss_fn(value.squeeze(), labels)  # Ignore -100
#         self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"]
        
#         value = self(input_ids, attention_mask)
        
#         loss = self.loss_fn(value.squeeze(), labels)
#         self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         return optimizer