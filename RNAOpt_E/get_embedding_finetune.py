from utils import RNADataset_finetune, RNATokenizer, load_config
from model import RNA_MaskedLM_finetune
from torch.utils.data import Dataset, DataLoader
import torch    
from tqdm import tqdm
import numpy as np
import torch.nn as nn



if __name__ == "__main__":
    # Load configuration
    config = load_config('/home/reagan/Projects/RNA_optimization/model/models_put_on_github/model_modules/configs/inference_config.yaml')

    # Initialize tokenizer
    tokenizer = RNATokenizer()

    # Create dataset and dataloader
    dataset = RNADataset_finetune(config['data_file'], tokenizer, config['max_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=48)

    # Initialize model
    model = RNA_MaskedLM_finetune(config=config)

    # Load pretrained weights
    checkpoint = torch.load(config['finetuned_model_path'])
    full_state_dict = checkpoint['state_dict']

    # torch.save(modified_state_dict, "14M_stage2_finetuned_on_human_mouse_new.ckpt")

    model.load_state_dict(full_state_dict, strict=True)

    model.eval().to("cuda:0")
    embedding_list = []
    label_list = []
    seq_gc_content_list = []
    seq_list = []
    pooling = nn.AdaptiveAvgPool1d(1)
    for batch in tqdm(dataloader):
        
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            seq = batch["seq"]
            
            _, _, last_hidden_state = model(input_ids, attention_mask)
            
            pooled_output = pooling(last_hidden_state.transpose(1, 2))  # Shape: [batch, hidden_dim, 1]
            pooled_output = pooled_output.squeeze(-1) # Shape: [batch, hidden_dim]
            
            #print(input_ids)
            print("last_hidden_state shape: ", pooled_output.shape)  # (batch_size, seq_len, hidden_size)
       
            embedding_list.extend(pooled_output.cpu().numpy())
            label_list.extend(labels.tolist())
            seq_list.extend(seq)
for seq in seq_list:
    # calculate GC content percentage
    gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100
    seq_gc_content_list.append(gc_content)



embeddings_array = np.array(embedding_list)  # shape: (N, D)
print("embeddings_array shape: ", embeddings_array.shape)
labels_array = np.array(label_list)          # shape: (N,)
print("labels_array shape: ", labels_array.shape)

gc_label_array = np.array(seq_gc_content_list)

# Save both into one file
np.savez("/home/reagan/Projects/RNA_optimization/model/models_put_on_github/UMAP_finetune/embeddings/all_UTR_data_embedding_with_labels.npz", embeddings=embeddings_array, hl_labels=labels_array, gc_labels=gc_label_array)
print("✅ Saved to embedding_with_labels.npz")
        