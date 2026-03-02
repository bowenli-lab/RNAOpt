from transformers import JambaModel, JambaConfig
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import pandas as pd
import yaml
import torch
import h5py

class RNATokenizer:
    def __init__(self):
        self.vocab = {"<CLS>": 0, "<SEP>": 1, "<UNK>": 2, "<PAD>": 3, "<MASK>": 4, "A": 5, "T": 6, "C": 7, "G": 8}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.mask_token_id = self.vocab["<MASK>"]  # ID for the mask token

    def tokenize(self, sequence):
        return ["<CLS>"] + list(sequence) + ["<SEP>"]

    def encode(self, sequence, max_length=None):
        tokens = self.tokenize(sequence)
        token_ids = [self.vocab.get(token, self.vocab["<PAD>"]) for token in tokens]
        if max_length is not None:
            token_ids = self.pad_sequence(token_ids, max_length)
        return token_ids

    def decode(self, token_ids):
        return "".join([self.inv_vocab[int(token_id)] for token_id in token_ids if int(token_id) in self.inv_vocab]), [self.inv_vocab[int(token_id)] for token_id in token_ids if int(token_id) in self.inv_vocab]

    def mask_tokens(self, inputs, mask_prob=0.15):
        """
        Randomly mask input tokens for masked language modeling.
        Unmasked tokens will have their labels set to -100.
        """
        labels = inputs.clone()
        mask = torch.rand(inputs.shape) < mask_prob

        # Do not mask special tokens ([CLS], [SEP], [PAD])
        special_token_mask = (inputs == self.vocab["<CLS>"]) | (inputs == self.vocab["<SEP>"]) | (inputs == self.vocab["<PAD>"])
        mask = mask & ~special_token_mask

        # Create indices for masking, keeping, and random replacement
        mask_indices = torch.where(mask)[0]
        num_mask = len(mask_indices)
        num_mask_80 = int(num_mask * 0.8)
        num_keep_10 = int(num_mask * 0.1)
        num_rand_10 = num_mask - num_mask_80 - num_keep_10  # Remaining 10%

        # Shuffle the mask indices to randomly distribute across the categories
        shuffled_indices = mask_indices[torch.randperm(num_mask)]

        # 80%: Replace with <MASK>
        mask_to_replace = shuffled_indices[:num_mask_80]
        inputs[mask_to_replace] = self.vocab["<MASK>"]

        mask_to_random = shuffled_indices[num_mask_80 + num_keep_10:]
        random_tokens = torch.randint(5, 9, (len(mask_to_random),), dtype=torch.long)
        inputs[mask_to_random] = random_tokens

        # Update labels: only masked tokens have valid labels
        labels[~mask] = -100  
        return inputs, labels

    def pad_sequence(self, token_ids, max_length):
        """
        Pads or truncates the token_ids to the specified max_length.
        """
        if len(token_ids) > max_length:
            return token_ids[:max_length]  # Truncate
        padding = [self.vocab["<PAD>"]] * (max_length - len(token_ids))
        return token_ids + padding  # Pad





class RNADataset_get_pretrain_embedding(Dataset):
    def __init__(self, hdf5_path, tokenizer, max_len):
        self.hdf5_path = hdf5_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.file = None

        # Load index only once
        with h5py.File(self.hdf5_path, "r") as h5f:
            self.length = len(h5f["data"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")

        seq = self.file["data"][idx].decode("utf-8")
        ids = self.file["ids"][idx].decode("utf-8")

        header = ids
        try:
            specie = header.split("|")[3].strip()
        except IndexError:
            print("Skipping header:", header)

        # Truncate if needed
        if len(seq) > self.max_len - 2:
            seq = seq[: self.max_len - 2]

        # Encode and mask
        input_ids = torch.tensor(self.tokenizer.encode(seq, max_length=self.max_len), dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.vocab["<PAD>"]).long()
        return input_ids, attention_mask, specie, seq

class RNADataset_finetune(Dataset):
    def __init__(self, csv_file, tokenizer, max_len, specie):
        self.sequences = self.load_csv(csv_file, max_len, specie)
        self.tokenizer = tokenizer
        self.max_len = max_len

    

    def load_csv(self, csv_file, max_len, specie):
        data = []
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        if specie == "Human":
            df = df[df["SPECIES"] == "Human"]
        elif specie == "Mouse":
            df = df[df["SPECIES"] == "Mouse"]
        elif specie == "All":
            df = df
        
        # Iterate through the rows of the DataFrame
        for _, row in df.iterrows():
            
            sequence = str(row['FULL_SEQUENCE']).upper()  # Ensure sequences are uppercase strings
            decay_rate = row['HALFLIFE']  # Extract decay_rate
            
            if len(sequence) > max_len - 2:  # Account for CLS and SEP tokens
                sequence = sequence[: max_len - 2]  # Truncate to fit within max_len
            
            # Append the sequence and decay_rate as a dictionary
            data.append((sequence, decay_rate))
        
        return data

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, decay_rate = self.sequences[idx]
        input_ids = torch.tensor(self.tokenizer.encode(sequence, max_length=self.max_len), dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.vocab["<PAD>"]).long()  # Mask for non-padding tokens
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": decay_rate, "seq": sequence}


class RNADataset_pretrain(Dataset):
    def __init__(self, hdf5_path, tokenizer, max_len, mask_prob):
        self.hdf5_path = hdf5_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.file = None

        # Load index only once
        with h5py.File(self.hdf5_path, "r") as h5f:
            self.length = len(h5f["data"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")
        seq = self.file["data"][idx].decode("utf-8")
        if len(seq) > self.max_len - 2:
            seq = seq[: self.max_len - 2]

        input_ids = torch.tensor(self.tokenizer.encode(seq, max_length=self.max_len), dtype=torch.long)

        attention_mask = (input_ids != self.tokenizer.vocab["<PAD>"]).long()
        input_ids, labels = self.tokenizer.mask_tokens(input_ids, mask_prob=self.mask_prob)

        return input_ids, attention_mask, labels

class RNADataset_search(Dataset):
    def __init__(self, seq, tokenizer, max_len, utr3=None, utr5=None):
        self.sequences = seq 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.utr3 = utr3
        self.utr5 = utr5


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence =  self.utr5 + self.sequences[idx] + self.utr3
        input_ids = torch.tensor(self.tokenizer.encode(sequence, max_length=self.max_len), dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.vocab["<PAD>"]).long()  # Mask for non-padding tokens
        return {"input_ids": input_ids, "attention_mask": attention_mask}




def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)


    if "JambaConfig" in raw_config:
        jamba_cfg = JambaConfig(**raw_config["JambaConfig"])
        raw_config["JambaConfig"] = jamba_cfg

    return raw_config

def load_config_distill_search(yaml_path):
    with open(yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if "JambaConfig" in raw_config:
        jamba_cfg = JambaConfig(**raw_config["JambaConfig"])
        raw_config["JambaConfig"] = jamba_cfg

    if "distill_JambaConfig" in raw_config:
        jamba_cfg = JambaConfig(**raw_config["distill_JambaConfig"])
        raw_config["distill_JambaConfig"] = jamba_cfg
    
    if "full_JambaConfig" in raw_config:
        jamba_cfg = JambaConfig(**raw_config["full_JambaConfig"])
        raw_config["full_JambaConfig"] = jamba_cfg

    return raw_config