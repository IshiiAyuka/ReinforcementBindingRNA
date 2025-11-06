import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import config

class RNADataset(Dataset):
    def __init__(self, protein_feat_file, csv_path, allowed_ids=None):
        full_feats_dict = torch.load(protein_feat_file)  # 例: {"2zni_A": tensor(...), ...}
        self.data = []
        self.ids = []

        df = pd.read_csv(csv_path, low_memory=False)

        for idx, row in df.iterrows():
            chain_id = str(row["subunit_1"]).strip() 
            complex_key = f"{chain_id}"
            rna_seq = str(row["s2_sequence"]).strip().upper()

            if allowed_ids is not None and complex_key not in allowed_ids:
                continue

            if not (len(rna_seq) <= 2050):
                continue

            if rna_seq == "NAN":
                continue

            if complex_key not in full_feats_dict:
                continue

            protein_feat = full_feats_dict[complex_key]

            tgt = torch.tensor([config.rna_vocab["<sos>"]] + [config.rna_vocab[c] for c in rna_seq] + [config.rna_vocab["<eos>"]], dtype=torch.long)
            self.data.append((protein_feat, tgt))
            #onlypredictのときのみ
            #self.ids.append(complex_key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        #onlypredictのときのみ
        '''protein_feat, tgt = self.data[idx]
        uid = self.ids[idx]          
        return protein_feat, tgt, uid'''

def custom_collate_fn(batch):
    protein_feats, tgt_seqs = zip(*batch)
    protein_feats = torch.stack(protein_feats)
    tgt_seqs = pad_sequence(tgt_seqs, batch_first=True, padding_value=config.rna_vocab["<pad>"])
    return protein_feats, tgt_seqs

def parse_clstr(clstr_path):
    clusters = []
    current_cluster = []
    with open(clstr_path, "r") as f:
        for line in f:
            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
            else:
                seq_id = line.strip().split(">")[1].split("...")[0]
                current_cluster.append(seq_id)
        if current_cluster:
            clusters.append(current_cluster)
    return clusters

class RNAOnlyDataset(Dataset):
    def __init__(self, protein_feat_path):
        self.data_dict = torch.load(protein_feat_path)
        self.keys = list(self.data_dict.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.data_dict[key], key 