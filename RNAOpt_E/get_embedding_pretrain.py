from utils import RNADataset_get_pretrain_embedding, RNATokenizer, load_config
from model import RNA_MaskedLM_pretrain
from torch.utils.data import Dataset, DataLoader
import torch    
from tqdm import tqdm
import numpy as np
import torch.nn as nn



if __name__ == "__main__":
    # Load configuration
    config = load_config('inference_config.yaml')

    # Initialize tokenizer
    tokenizer = RNATokenizer()

    # Create dataset and dataloader
    dataset = RNADataset_get_pretrain_embedding(config['pretrain_data_file'], tokenizer, config['max_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=48)

    # Initialize model
    model = RNA_MaskedLM_pretrain(config=config)
    model.eval().to("cuda:0")  # Set to eval mode and move to GPU
    

    # model parallel for multi-gpu inference

    # Load pretrained weights
    checkpoint = torch.load(config['pretrained_model_path'])
    full_state_dict = checkpoint['state_dict']
    modified_state_dict = {}
    for key, value in full_state_dict.items():
        # Modify the key to match the model's expected state_dict keys
        new_key = key.replace("model.model.model", "model.model")
        modified_state_dict[new_key] = value
    # torch.save(modified_state_dict, "14M_stage2_finetuned_on_human_mouse_new.ckpt")

    model.load_state_dict(modified_state_dict, strict=True)

    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    
    embedding_list = []
    label_list = []
    seq_list = []
    pooling = nn.AdaptiveAvgPool1d(1)
    for batch in tqdm(dataloader):
        
        with torch.no_grad():
            input_ids, attention_mask, specie, seq = batch
            input_ids = input_ids.to("cuda:0")
            attention_mask = attention_mask.to("cuda:0")
            
            logits, last_hidden_state = model(input_ids, attention_mask)
            
            pooled_output = pooling(last_hidden_state.transpose(1, 2))  # Shape: [batch, hidden_dim, 1]
            pooled_output = pooled_output.squeeze(-1) # Shape: [batch, hidden_dim]
            
            #print(input_ids)
            print("last_hidden_state shape: ", pooled_output.shape)  # (batch_size, seq_len, hidden_size)
       
            embedding_list.extend(pooled_output.cpu().numpy())
            label_list.extend(specie)
            seq_list.extend(seq)

seq_list_gc_content = []    
for seq in seq_list:
    # calculate GC content percentage
    gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100
    seq_list_gc_content.append(gc_content)


embeddings_array = np.array(embedding_list)  # shape: (N, D)
print("embeddings_array shape: ", embeddings_array.shape)
labels_array = seq_list_gc_content         # shape: (N,)


print("labels_array shape: ", len(labels_array))

# Save both into one file
np.savez("all_UTR_data_embedding_with_gccontent_labels.npz", embeddings=embeddings_array, labels=labels_array)
print("✅ Saved to embedding_with_labels.npz")
        