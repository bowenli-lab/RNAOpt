from utils import RNADataset_finetune, RNATokenizer, load_config
from model import RNA_MaskedLM_finetune
from torch.utils.data import Dataset, DataLoader
import torch    
from tqdm import tqdm
import pandas as pd
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

    # Initialize tokenizer
    tokenizer = RNATokenizer()

    # Create dataset and dataloader
    dataset = RNADataset_finetune(config['data_file'], tokenizer, config['max_len'], specie="Human")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=48)

    # Initialize model
    model = RNA_MaskedLM_finetune(config=config)

    # Load pretrained weights
    checkpoint = torch.load(config['finetuned_model_path'])
    full_state_dict = checkpoint['state_dict']


    model.load_state_dict(full_state_dict, strict=True)

    model.eval().to("cuda:0")
    all_prediction_list = []
    all_label_list = []
    for batch in tqdm(dataloader):
        
        with torch.no_grad():
            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            
            value, _, _ = model(input_ids, attention_mask)

       
            all_prediction_list.extend([ele[0] for ele in value.tolist()])
            all_label_list.extend(labels.tolist())

    import scipy.stats as stats

    # Example data
    A = all_label_list  
    B = all_prediction_list 

    # Calculate Spearman's rank correlation coefficient
    spearman_corr, p_value = stats.spearmanr(A, B)
    pearson_corr, p_value2 = stats.pearsonr(A, B)


    print(f"Pearson's Correlation: {pearson_corr:.4f}")
    print(f"P-value: {p_value2:.4f}")

    print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
    print(f"P-value: {p_value:.4f}")
        