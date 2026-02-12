import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
#import Decoder.config as config
import config 

class RNADataset_AR(Dataset):
    def __init__(self, protein_feat_file, csv_path, allowed_ids=None):
        full_feats_dict = torch.load(protein_feat_file)  
        self.data = []

        df = pd.read_csv(csv_path, low_memory=False)

        for _, row in df.iterrows():
            chain_id = str(row["subunit_1"]).strip()
            uid = chain_id
            rna_seq = str(row["s2_sequence"]).strip().upper()
            prot_seq = str(row["s1_sequence"]).strip().upper() 

            if allowed_ids is not None and uid not in allowed_ids:
                continue
            if rna_seq == "NAN":
                continue
            if not (config.min_len <= len(rna_seq) <= config.max_len - 2):
                continue
            if uid not in full_feats_dict:
                continue

            vocab = config.rna_vocab
            try:
                tok_body = [vocab[c] for c in rna_seq]  
            except KeyError:
                continue

            tgt = torch.tensor(
                [vocab["<sos>"]] + tok_body + [vocab["<eos>"]],
                dtype=torch.long
            )
            protein_feat = torch.as_tensor(full_feats_dict[uid]).float()

            self.data.append((protein_feat, tgt, prot_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_feat, tgt, prot_seq = self.data[idx]
        return protein_feat, tgt, prot_seq

def custom_collate_fn_AR(batch):
    pad_id = config.rna_vocab["<pad>"]

    protein_feats, tgt_seqs, prot_seqs = zip(*batch) 

    B = len(protein_feats)
    D = protein_feats[0].size(1)
    S_max = max(feat.size(0) for feat in protein_feats)

    protein_batch = torch.zeros(B, S_max, D, dtype=torch.float32)
    for i, feat in enumerate(protein_feats):
        S_i = feat.size(0)
        protein_batch[i, :S_i] = feat

    # --- RNA ターゲット: PAD で右詰めパディング ---
    maxL = max(t.size(0) for t in tgt_seqs)
    tgt_padded = torch.full((B, maxL), pad_id, dtype=torch.long)
    for i, t in enumerate(tgt_seqs):
        L = t.size(0)
        tgt_padded[i, :L] = t

    return protein_batch, tgt_padded, list(prot_seqs)

    
class RNADataset_deepclip_AR(torch.utils.data.Dataset):
    def __init__(self, protein_feat_file, csv_path, allowed_ids=None):
        full_feats_dict = torch.load(protein_feat_file)
        self.data = []

        df = pd.read_csv(csv_path, low_memory=False)
        for _, row in df.iterrows():
            file_name = str(row["file_name"]).strip()
            prot_seq = str(row["sequence"]).strip().upper()

            if allowed_ids is not None and file_name not in allowed_ids:
                continue
            if prot_seq == "NAN" or prot_seq == "":
                continue
            if file_name not in full_feats_dict:
                continue

            protein_feat = torch.as_tensor(full_feats_dict[file_name]).float()

            self.data.append((protein_feat, prot_seq, file_name))

        print(f"[DeepCLIPProteinDataset] 有効 {len(self.data)} 件")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_feat, prot_seq, file_name = self.data[idx]
        return protein_feat, prot_seq, file_name

def custom_collate_fn_deepclip_AR(batch):
    protein_feats, prot_seqs, file_names = zip(*batch)

    B = len(protein_feats)
    D = protein_feats[0].size(1)
    S_max = max(feat.size(0) for feat in protein_feats)

    protein_batch = torch.zeros(B, S_max, D, dtype=torch.float32)
    for i, feat in enumerate(protein_feats):
        S_i = feat.size(0)
        protein_batch[i, :S_i] = feat

    return protein_batch, list(prot_seqs), list(file_names)


def read_fasta_ids_and_seqs(fasta_path: str):
    ids, seqs = [], []
    cur_id, buf = None, []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(buf).replace(" ", "").upper())
                header = line[1:].strip()
                parts = header.split("|")
                cur_id = parts[1] if len(parts) >= 2 and parts[0] in ("sp", "tr") else header.split()[0]
                buf = []
            else:
                buf.append(line)
    if cur_id is not None:
        ids.append(cur_id)
        seqs.append("".join(buf).replace(" ", "").upper())
    return ids, seqs


class ProteinFeatFastaDictDataset(Dataset):
    """
    .pt が dict: {protein_id: featTensor} のときのDataset
    featTensor は [S,D] or [1,S,D] or [D] を許容
    """
    def __init__(self, protein_feat_pt_path: str, fasta_path: str):
        obj = torch.load(protein_feat_pt_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise TypeError(f".ptの中身がdictではありません: {type(obj)}")
        self.feat_dict = obj
        self.ids, self.seqs = read_fasta_ids_and_seqs(fasta_path)

        # 存在チェック（最初の数件だけでもOKだが、ここでは全件チェック）
        missing = [pid for pid in self.ids if pid not in self.feat_dict]
        if len(missing) > 0:
            # よくあるsp|acc|name問題の可能性があるので、例を出す
            example_keys = list(self.feat_dict.keys())[:5]
            raise KeyError(f"FASTA IDが.ptに見つかりません（例: {missing[:5]}）。ptキー例: {example_keys}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        return pid, self.seqs[idx]


def custom_collate_fn_feat_fasta_dict(batch, feat_dict: dict):
    """
    戻り値は (protein_feat [B,S,D], None, protein_seq_list)
    """
    pids = [x[0] for x in batch]
    protein_seq_list = [x[1] for x in batch]

    feats = []
    for pid in pids:
        x = feat_dict[pid]
        if torch.is_tensor(x):
            if x.dim() == 3 and x.size(0) == 1:
                x = x.squeeze(0)
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() != 2:
                raise ValueError(f"feat dim must be 1/2/3(1,S,D), got {tuple(x.shape)} for pid={pid}")
            feats.append(x.to(torch.float32))
        else:
            raise TypeError(f"feat_dict[{pid}] is not tensor: {type(x)}")

    protein_feat = pad_sequence(feats, batch_first=True)  # [B,S,D]
    return protein_feat, None, protein_seq_list