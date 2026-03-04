
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))




sys.path.append(os.path.join(BASE_DIR, "..", "RNAOpt_E"))

from model import RNA_MaskedLM_finetune, RNA_MaskedLM_pretrain
from utils import RNATokenizer, RNADataset_search, load_config, load_config_distill_search

import heapq
from Bio.Data import CodonTable
from Bio.Seq import Seq
import torch
import torch.nn as nn
from transformers import JambaModel, JambaConfig
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import lightning as pl

from lightning.pytorch.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger  
# import wandb
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import pandas as pd
import pickle
import random
from tqdm import tqdm

import random
import numpy as np
import time
import torch

from CAI import CAI
from collections import defaultdict
import pandas as pd
from Bio.Seq import Seq
import argparse







# --- Load codon frequency table ---
human_codon_freq_path = os.path.join(BASE_DIR, "human_codon_freq_table.txt")
raw_usage = {}
with open(human_codon_freq_path, 'r') as f:
    for line in f:
        tokens = line.strip().split()
        for i in range(0, len(tokens), 5):
            if i + 2 >= len(tokens):
                continue
            rna_codon = tokens[i]
            freq = float(tokens[i + 2])
            dna_codon = rna_codon.replace('U', 'T')
            raw_usage[dna_codon] = freq

aa_to_codons = defaultdict(list)
for codon, freq in raw_usage.items():
    try:
        aa = str(Seq(codon).translate())
        aa_to_codons[aa].append((codon, freq))
    except Exception:
        continue

HUMAN_CODON_USAGE = {}
for aa, codon_freqs in aa_to_codons.items():
    max_freq = max(freq for _, freq in codon_freqs)
    for codon, freq in codon_freqs:
        HUMAN_CODON_USAGE[codon] = freq / max_freq


# print(HUMAN_CODON_USAGE)

# Build reverse mapping from AA to least-used codon
AA_TO_LEAST_CODON = {}

# Group codons by amino acid
aa_to_codons = defaultdict(list)
for codon, rel_freq in HUMAN_CODON_USAGE.items():
    try:
        aa = str(Seq(codon).translate())
        if aa == '*':  # skip stop codons unless needed
            continue
        aa_to_codons[aa].append((codon, rel_freq))
    except Exception:
        continue

# For each amino acid, select the codon with minimum relative frequency
for aa, codon_list in aa_to_codons.items():
    least_codon = min(codon_list, key=lambda x: x[1])[0]
    AA_TO_LEAST_CODON[aa] = least_codon


def translate_codon(codon_str):
    return str(Seq(codon_str.replace('U', 'T')).translate())
    
def set_seed(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def back_translate_least_frequent(protein_seq, aa_to_least_codon):
    protein_seq = protein_seq.upper()
    mrna_seq = ""
    for aa in protein_seq:
        if aa not in aa_to_least_codon:
            raise ValueError(f"No codon found for amino acid: {aa}")
        codon = aa_to_least_codon[aa].replace("U", "T")  # Use RNA
        mrna_seq += codon
    return mrna_seq


def calc_CAI(seq):
    seq_dna = seq.replace('U', 'T')  # just in case
    cai = CAI(seq_dna, HUMAN_CODON_USAGE)
    return cai


def score_function_CAI_filter(sequences, max_len=None, model=None, tokenizer=None, batch_size=32, utr3=None, utr5=None, theta=None):
    # Use our model to predict the half-life values for the given sequence
    torch.set_float32_matmul_precision("medium")
    
    # Prepare dataset and dataloader
    dataset = RNADataset_search(sequences, tokenizer, max_len=max_len, utr3=utr3, utr5=utr5)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

    
    model.eval()
    all_predictions = []
    all_CAIs = []
    with torch.no_grad():
        for batch in data_loader:
            
            input_ids = batch["input_ids"].to("cuda")  # Move to GPU
            # print("input_ids: ", input_ids)
            attention_mask = batch["attention_mask"].to("cuda")  # Move to GPU
            
            value, _, _ = model(input_ids, attention_mask)
            all_predictions.append(value)
              # Assuming batch["sequence"] contains the original sequences
    
    for seq in sequences:
        cai = calc_CAI(seq)
        all_CAIs.append(cai)
    
    all_predictions = torch.cat(all_predictions, dim=0).to("cpu")  # Concatenates along batch dimension
    all_CAIs = torch.tensor(all_CAIs, dtype=torch.float32).unsqueeze(1)  # Convert to tensor and add dimension

    merged_scores = all_predictions + theta * all_CAIs  # Merging the scores

    return all_predictions, all_CAIs, merged_scores  






def get_possible_moves(sequence, visited_sequences): # visited_sequences is a set
        codon_table = CodonTable.unambiguous_dna_by_id[1]  # Standard Genetic Code

        possible_moves = []

        # Process codons in triplets
        for i in range(0, len(sequence) - 2, 3):  
            codon = sequence[i:i+3]  # Extract the codon
            amino_acid = Seq(codon).translate()  # Translate codon to amino acid

            for j in range(3):  # Iterate through each nucleotide in the codon
                for nucleotide in "ATCG":
                    if nucleotide != codon[j]:  # Avoid replacing with the same nucleotide
                        new_codon = codon[:j] + nucleotide + codon[j+1:]  # Create new codon
                        new_amino_acid = Seq(new_codon).translate()  # Translate the new codon

                        if new_amino_acid == amino_acid:  # Check if mutation is synonymous
                            new_sequence = sequence[:i+j] + nucleotide + sequence[i+j+1:]  # Apply mutation
                            
                            if new_sequence not in visited_sequences:  # Prevent revisiting
                                # visited_sequences.add(new_sequence)  
                                possible_moves.append(new_sequence)

        return possible_moves


def remove_bsmBI_bspQI(sequences):
    list_motif_to_remove = ["GCTCTTC", "GAAGAGC", "CGTCTC", "GAGACG"]
    clean_sequences = [
        seq for seq in sequences 
        if not any(motif in seq for motif in list_motif_to_remove)
    ]

    return clean_sequences
                
            





def CoT_distill_CAI_filter(start_sequence, beam_width=10, max_steps=10, max_len=3072, batch_size=16, utr5=None, utr3=None, distill_model_config=None, distill_model_path=None, full_model_path=None, full_model_config=None, beta=None, theta=None, patience=None, partial_sampling_percentage=None):
    
    # Initialize tokenizer & model
    tokenizer = RNATokenizer()
    vocab = tokenizer.vocab
    distill_model = RNA_MaskedLM_finetune(config=distill_model_config)

    distill_checkpoint = torch.load(distill_model_path, map_location="cpu")

    distill_full_state_dict = distill_checkpoint['state_dict']

    distill_model.load_state_dict(distill_full_state_dict, strict=True)
    distill_model = distill_model.to(torch.bfloat16)
    
    distill_model.eval().to("cuda")  # Set to eval mode and move to GPU

    full_model = RNA_MaskedLM_finetune(config=full_model_config)
    full_model_checkpoint = torch.load(full_model_path, map_location="cpu", weights_only=False)

    full_model_full_state_dict = full_model_checkpoint['state_dict']


    full_model.load_state_dict(full_model_full_state_dict, strict=True)

    full_model = full_model.to(torch.bfloat16)
    full_model.eval().to("cuda")

    visited_sequences = set()
    
    top_sequences = []  
    top_sequences_scores_merged = []
    top_sequences_hls = []
    top_sequences_CAIs = []
    
    
    # Initialize the beam with the starting sequence and its score

    initial_score_tensor, initial_CAI_tensor, merged_score_tensor = score_function_CAI_filter([start_sequence], max_len=max_len, model=full_model, tokenizer=tokenizer, utr3=utr3, utr5=utr5, theta=theta)

    initial_hl = initial_score_tensor.item()
    initial_CAI = initial_CAI_tensor.item()
    initial_merged_score = merged_score_tensor.squeeze(0).item()  # Remove batch dimension if present
    
    print("Initial Score: ", initial_hl)
    print("Initial CAI: ", initial_CAI)
    print("Initial Fitness Socre: ", initial_merged_score)

    
    beam = [start_sequence]  # Initialize beam with the initial sequence and its score

    
    visited_sequences.add(start_sequence)  # Store in a hashable form

    print("Initial Sequence: ", start_sequence)

    
    all_search_start_time = time.time()
    best_hl = initial_hl
    best_CAI = initial_CAI
    best_merged_metric = initial_merged_score
    
    best_hl_tracker = []
    best_CAI_tracker = []
    best_merged_metric_tracker = []
    
    # time_stamp_tracker = []
    # step_tracker = []
    beam_tracker = {}

    no_improvement_counter = 0

    for step in tqdm(range(max_steps)):
        if no_improvement_counter >= patience:
            print(f"No improvement for {patience} steps, stopping search.")
            break
        all_candidates = []
        all_candidates_hl = []
        all_candidates_CAI = []
        all_candidates_merged_metrics = []

        samples_direct_to_full_prediction = []
        
        for seq in beam:
            # print("Number of visited sequences: ", len(visited_sequences))
            
            successors_original_o = get_possible_moves(seq, visited_sequences)  # Generate possible next sequences
            # print("Number of possible moves: ")
            # print(len(successors_original_o))

            

            successors_original = remove_bsmBI_bspQI(successors_original_o)

            if len(successors_original) == 0:
              continue

            random.shuffle(successors_original)

            num_samples = int(partial_sampling_percentage * len(successors_original))  # Sample a fraction of successors
            # print(f"Number of successors for sequence: {num_samples}")
            successors = successors_original[:num_samples]  # Limit the number of successors to 100 for efficiency
            seq_dir_to_full_prediction = successors_original[num_samples:num_samples + beta]  # Keep the original sequence for direct full model prediction
            # print(seq_dir_to_full_prediction)
            samples_direct_to_full_prediction.extend(seq_dir_to_full_prediction)  # Sample a fraction of successors for direct full model prediction

            
            
            # print("Number of possible moves: ", len(successors))

            _, _, merged_metric_values = score_function_CAI_filter(successors, max_len=max_len, model=distill_model, tokenizer=tokenizer, batch_size=batch_size, utr3=utr3, utr5=utr5, theta=theta)  # Evaluate new sequence
                
                
            for i, seq in enumerate(successors):
                
                all_candidates.append(seq)
                # all_candidates_hl.append(hl[i].item())
                # all_candidates_logits.append(logits[i].item())
                all_candidates_merged_metrics.append(merged_metric_values[i].item())
                # print("length of all_candidates_merged_metrics: ", len(all_candidates_merged_metrics))
                visited_sequences.add(seq)

            for i, seq in enumerate(seq_dir_to_full_prediction):
                visited_sequences.add(seq)

        
        combined = list(zip(all_candidates_merged_metrics, all_candidates))
        beam_wide = heapq.nlargest(beam_width * beta, combined, key=lambda x: x[0])

        seqs = [seq for _, seq in beam_wide]
        seqs.extend(samples_direct_to_full_prediction)  # Add the sampled sequences for direct full model prediction

        all_hls, all_CAIs, all_merged_metrics = score_function_CAI_filter(seqs, max_len=max_len, model=full_model, tokenizer=tokenizer, batch_size=batch_size, utr3=utr3, utr5=utr5, theta=theta)

        best_index_merged = torch.argmax(all_merged_metrics)
        best_hl_s = all_hls[best_index_merged].item()
        best_CAI_s = all_CAIs[best_index_merged].item()
        best_merged_metric_s = all_merged_metrics[best_index_merged].item()

        if best_merged_metric_s > best_merged_metric:
            best_merged_metric = best_merged_metric_s
            best_hl = best_hl_s
            best_CAI = best_CAI_s
            no_improvement_counter = 0
            print(f"New Best Merged Metric: {best_merged_metric}, Best HL: {best_hl}, Best CAI: {best_CAI}")
        else:
            no_improvement_counter += 1
            
        
        combiend_candidates = list(zip(all_hls.squeeze().tolist(), all_CAIs.squeeze().tolist(), all_merged_metrics.squeeze().tolist(), seqs))
        beam_inter = heapq.nlargest(beam_width, combiend_candidates, key=lambda x: x[2])

        beam = []
        for hl, cai, merged_metric, seq in beam_inter:
            top_sequences.append(seq)
            top_sequences_hls.append(hl)
            top_sequences_CAIs.append(cai)
            top_sequences_scores_merged.append(merged_metric)
            beam.append(seq)

        for seq in beam:
            beam_tracker[seq] = step

        # save tracker to csv
        
        step_list = []
        for seq_d in top_sequences:
            step_d = beam_tracker[seq_d]
            step_list.append(step_d)
        # df_top = pd.DataFrame({"Top_Sequence": top_sequences, "Top_HL": top_sequences_hls, "Top_CAIs": top_sequences_CAIs, "Top_Score_Merged": top_sequences_scores_merged, "Step": step_list})
        # df_top.to_csv("/home/reagan/Projects/rnaopt/CoT/formal_optimization/CAR_T/CAR_sequences_distill_merged_metric_20CAI_CAR_interfile.csv")
   
    return top_sequences, top_sequences_hls, top_sequences_CAIs, top_sequences_scores_merged, beam_tracker #, torch.tensor(all_best_scores_tracker).cpu().tolist(), torch.tensor(time_stamp_tracker).cpu().tolist(), torch.tensor(step_tracker).cpu().tolist()  # Return top 10 sequences and their scores













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
        config = load_config_distill_search(args.config)
        print(f"Configuration loaded successfully from: {args.config}")
    else:
        raise FileNotFoundError(f"Config file not found at: {args.config}")
    print("Configuration loaded successfully.")

    set_seed(config['seed'])
    if config["start_from_protein"] == True:
        input_protein_sequence = config["input_protein_sequence"]
        cds = back_translate_least_frequent(input_protein_sequence, AA_TO_LEAST_CODON) + "TGA"
    
    else:
        cds = config["cds"].upper()  # Convert to uppercase to ensure consistency
    
    utr5 = config["utr5"]
    utr3 = config["utr3"]

    start_sequence = cds + utr5 + utr3  # Concatenate CDS, UTR5, and UTR3 sequences
    
    max_len = len(start_sequence) + 2
    # max_steps = max_len
    print("Start Sequence: ", cds)
    # print("Max Steps: ", max_steps)

    start_time = time.time()

    distill_model_config = {
        "learning_rate": 0,
        "vocab_size": config["vocab_size"],
        
        "JambaConfig": config['distill_JambaConfig'],
        
        }

    full_model_config = {   
        "learning_rate": 0,
        "warmup_steps": 0,
        'total_steps': 0,
        "vocab_size": config["vocab_size"],
        "JambaConfig": config['full_JambaConfig'],
        }
    top_sequences, top_sequences_hls, top_sequences_CAIs, top_sequences_scores_merged, beam_tracker = CoT_distill_CAI_filter(cds, 
                                                                                                    beam_width=config["beam_width"], 
                                                                                                    max_steps=config['max_steps'], 
                                                                                                    batch_size=config["batch_size"], 
                                                                                                    max_len=max_len, 
                                                                                                    distill_model_path=config["distill_model_path"],
                                                                                                    full_model_path=config["full_model_path"],
                                                                                                    distill_model_config=distill_model_config,
                                                                                                    full_model_config=full_model_config,
                                                                                                    utr5=utr5,
                                                                                                    utr3=utr3,
                                                                                                    beta=config["beta"],
                                                                                                    theta=config["theta"],
                                                                                                    patience=config["patience"],
                                                                                                    partial_sampling_percentage=config["partial_sampling_percentage"]
                                                                                                    )

    

    end_time = time.time()
    print("Total Time: ", end_time - start_time)

    # print("Top Sequence: ", top_sequences)
    # save top_sequences to a csv file
    step_list = []
    for seq in top_sequences:
        step = beam_tracker[seq]
        step_list.append(step)

    df_top = pd.DataFrame({"Top_Sequence": top_sequences, "Top_HL": top_sequences_hls, "Top_CAIs": top_sequences_CAIs, "Top_Score_Merged": top_sequences_scores_merged, "Step": step_list})
    df_top.to_csv(config["result_save_path"])

   