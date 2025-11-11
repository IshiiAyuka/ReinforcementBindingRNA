import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
#import Decoder.config as config
import config 

class RNADataset_AR(Dataset):
    def __init__(self, protein_feat_file, csv_path, allowed_ids=None):
        full_feats_dict = torch.load(protein_feat_file)  # 例: {"2zni_A": tensor(...), ...}
        self.data = []
        self.ids = []

        df = pd.read_csv(csv_path, low_memory=False)

        for _, row in df.iterrows():
            chain_id = str(row["subunit_1"]).strip()
            uid = f"{chain_id}"
            rna_seq = str(row["s2_sequence"]).strip().upper()
            prot_seq = str(row["s1_sequence"]).strip().upper() 

            # フィルタ
            if allowed_ids is not None and uid not in allowed_ids:
                continue
            if rna_seq == "NAN":
                continue
            # <sos>, <eos> を含めて max_len に収まるように
            if not (config.min_len <= len(rna_seq) <= config.max_len - 2):
                continue
            if uid not in full_feats_dict:
                continue

            vocab = config.rna_vocab
            try:
                tok_body = [vocab[c] for c in rna_seq]  # A/U/C/G のみ想定
            except KeyError:
                # 予期しない文字（Nなど）が含まれる行はスキップ
                continue

            tgt = torch.tensor(
                [vocab["<sos>"]] + tok_body + [vocab["<eos>"]],
                dtype=torch.long
            )
            protein_feat = torch.as_tensor(full_feats_dict[uid])

            self.data.append((protein_feat, tgt, prot_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_feat, tgt, prot_seq = self.data[idx]
        return protein_feat, tgt, prot_seq


def custom_collate_fn(batch):
    pad_id = config.rna_vocab["<pad>"]

    protein_feats, tgt_seqs, prot_seqs = zip(*batch)  # [(D,), (T,), str] × B

    # [B, D]
    protein_feats = torch.stack([torch.as_tensor(x) for x in protein_feats], dim=0).float()

    # PAD で右詰めパディング
    maxL = max(t.size(0) for t in tgt_seqs) if len(tgt_seqs) > 0 else 0
    B = len(tgt_seqs)
    tgt_padded = torch.full((B, maxL), pad_id, dtype=torch.long)
    for i, t in enumerate(tgt_seqs):
        L = t.size(0)
        tgt_padded[i, :L] = t

    return protein_feats, tgt_padded, list(prot_seqs)

class RNADataset_NAR(Dataset):
    def __init__(self, protein_feat_file, csv_path, allowed_ids=None):
        full_feats_dict = torch.load(protein_feat_file)  # 例: {"2zni_A": tensor(...), ...}
        self.data = []
        #self.ids = []

        df = pd.read_csv(csv_path, low_memory=False)

        for _, row in df.iterrows():
            chain_id = str(row["subunit_1"]).strip() 
            uid = f"{chain_id}"
            rna_seq = str(row["s2_sequence"]).strip().upper()
            #DeepCLIPのときは以下を削除
            prot_seq = str(row["s1_sequence"]).strip().upper()

            if allowed_ids is not None and uid not in allowed_ids:
                continue
            if not (config.min_len <= len(rna_seq) <= config.max_len - 2):
                continue

            if rna_seq == "NAN":
                continue
            if uid not in full_feats_dict:
                continue

            protein_feat = full_feats_dict[uid]

            tgt = torch.tensor([config.rna_vocab_NAR["<sos>"]] + [config.rna_vocab_NAR[c] for c in rna_seq] + [config.rna_vocab_NAR["<eos>"]], dtype=torch.long)
            self.data.append((protein_feat, tgt, prot_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #return self.data[idx]
        #onlypredictのときのみ
        protein_feat, tgt, prot_seq = self.data[idx]          
        return protein_feat, tgt, prot_seq

def custom_collate_fn(batch):
    """
    NAR（非逐次, 全<MASK>入力）用の最小戻り値版。
    戻り値（3要素）:
      - protein_feats [B, D]
      - inputs_parallel [B, L]  ← すべて <MASK>（ダミーOK）
      - prot_seqs (list[str])   ← そのまま（報酬モデルに渡す用）
    ※ L は min(バッチ内最大長, config.max_len) に頭打ち
    """
    nar = config.rna_vocab_NAR
    mask_id = nar["<MASK>"]

    protein_feats, tgt_seqs, prot_seqs = zip(*batch)  # [(D,), (T,), str] × B

    # [B, D]
    protein_feats = torch.stack([torch.as_tensor(x) for x in protein_feats], dim=0).float()

    # <sos> を除いた長さで L を決定
    seqs_wo_sos = [t[1:].long() for t in tgt_seqs]
    maxL_batch = max(s.size(0) for s in seqs_wo_sos) if seqs_wo_sos else 0
    L = min(maxL_batch, config.max_len) if maxL_batch > 0 else 0
    B = len(seqs_wo_sos)

    # 入力は全<MASK>（モデル側で forward_parallel(out_len=...) を使うなら実際は未使用でもOK）
    inputs_parallel = (
        torch.full((B, L), mask_id, dtype=torch.long) if L > 0
        else torch.empty((B, 0), dtype=torch.long)
    )

    return protein_feats, inputs_parallel, list(prot_seqs)

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
    
# ========= CSV 専用 Dataset と collate =========
class DeepCLIPProteinDataset(torch.utils.data.Dataset):
    def __init__(self, feat_pt_path: str, csv_path: str,
                 id_priority=("protein_name", "file_name"),
                 encoding: str = None):
        super().__init__()
        enc = encoding if encoding is not None else "cp932"
        self.df = pd.read_csv(csv_path, low_memory=False, encoding=enc)
        self.feat_dict = torch.load(feat_pt_path, map_location="cpu")

        self.items = []  # List[Tuple[key(str), prot_seq(str)]]
        skipped = 0
        for _, row in self.df.iterrows():
            fname = str(row.get("file_name", "")).strip()
            pname = str(row.get("protein_name", "")).strip()

            parts = [x for x in [fname, pname] if x]
            key = "_".join(parts) if parts else "unknown"

            # .pt に存在しないキーはスキップ
            if key not in self.feat_dict:
                skipped += 1
                continue

            prot_seq = str(row.get("sequence", "")).strip()
            if not prot_seq:
                skipped += 1
                continue

            self.items.append((key, prot_seq))

        if len(self.items) == 0:
            raise ValueError("特徴量に一致する行が0件でした。CSVの file_name / protein_name と .pt のキー対応を確認してください。")

        print(f"[DeepCLIPProteinDataset] 有効 {len(self.items)} 件 / スキップ {skipped} 件")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        key, prot_seq = self.items[idx]
        feat = self.feat_dict[key].float()
        return feat, key, prot_seq, None, None

class FastaProteinDataset(Dataset):
    """
    FASTAと特徴量.pt({acc: Tensor([D]) or Tensor([N,D])})を突き合わせるDataset
    戻り値: (protein_feat[D], protein_seq:str)
    """
    def __init__(self, feat_pt_path: str, fasta_path: str, pool: str = "mean", skip_missing: bool = True):
        super().__init__()
        self.items = []
        self.pool = pool

        feat_dict = torch.load(feat_pt_path, map_location="cpu")  # {acc: Tensor or array}
        # 期待次元（存在すればチェックする）
        expected_D = getattr(config, "input_dim", None)

        hit = miss = 0
        for header, prot_seq in self._iter_fasta(fasta_path):
            acc = self._acc_from_header(header)

            # --- キーフォールバック（アイソフォームなど） ---
            feat = self._get_feat_with_fallback(feat_dict, acc)

            if feat is None:
                miss += 1
                if skip_missing:
                    continue
                raise KeyError(f"Feature not found for accession: {acc}")

            # dtype/shapeを明示して安定化
            x = torch.as_tensor(feat, dtype=torch.float32)
            if x.ndim == 2:                          # [N, D] → [D]
                x = x.mean(dim=0) if self.pool == "mean" else x[0]
            elif x.ndim != 1:
                raise ValueError(f"Feature for {acc} must be 1D or 2D, got shape {tuple(x.shape)}")

            # 次元チェック（期待次元がわかる場合のみ）
            if expected_D is not None and x.numel() != expected_D:
                raise ValueError(f"Dim mismatch for {acc}: got {x.numel()}, expected {expected_D}")

            self.items.append((x, prot_seq))
            hit += 1

        print(f"[FastaProteinDataset] matched={hit}, missing_features={miss}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]  # (feat[D], prot_seq:str)

    # ---- 内部ユーティリティ ----
    @staticmethod
    def _acc_from_header(h: str) -> str:
        """
        例: '>sp|A0A075QQ08-2|IF4E1_TOBAC ...' → 'A0A075QQ08-2'
        """
        h0 = h.strip().split()[0]
        if h0.startswith('>'):
            h0 = h0[1:]
        parts = h0.split('|')
        return parts[1] if len(parts) >= 2 else parts[0]

    @staticmethod
    def _iter_fasta(path: str):
        head, seq_parts = None, []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if head is not None:
                        yield head, ''.join(seq_parts).upper()
                    head, seq_parts = line[1:].strip(), []
                else:
                    s = line.strip()
                    if s:  # 空行スキップ
                        seq_parts.append(s)
        if head is not None:
            yield head, ''.join(seq_parts).upper()

    @staticmethod
    def _get_feat_with_fallback(feat_dict: dict, acc: str):
        feat = feat_dict.get(acc)
        if feat is None and "-" in acc:
            base = acc.split("-", 1)[0]
            feat = feat_dict.get(base)
        return feat


def collate_rl(batch):
    feats, prot_seqs = zip(*batch)  # list[tensor[D]], list[str]
    feats = torch.stack(
        [b if isinstance(b, torch.Tensor) and b.dtype == torch.float32
         else torch.as_tensor(b, dtype=torch.float32)
         for b in feats],
        dim=0
    )
    return feats, None, list(prot_seqs)

class RNADataset_decodertrain(Dataset):
    def __init__(self, protein_feat_file, csv_path, allowed_ids=None):
        full_feats_dict = torch.load(protein_feat_file)  # 例: {"2zni_A": tensor(...), ...}
        self.data = []
        #self.ids = []

        df = pd.read_csv(csv_path, low_memory=False)

        for _, row in df.iterrows():
            chain_id = str(row["subunit_1"]).strip() 
            uid = f"{chain_id}"
            rna_seq = str(row["s2_sequence"]).strip().upper()
            #DeepCLIPのときは以下を削除
            prot_seq = str(row["s1_sequence"]).strip().upper()

            if allowed_ids is not None and uid not in allowed_ids:
                continue
            if not (len(rna_seq) <= 502):
                continue
            if rna_seq == "NAN":
                continue
            if uid not in full_feats_dict:
                continue

            protein_feat = full_feats_dict[uid]

            tgt = torch.tensor([config.rna_vocab_NAR["<sos>"]] + [config.rna_vocab_NAR[c] for c in rna_seq] + [config.rna_vocab_NAR["<eos>"]], dtype=torch.long)
            self.data.append((protein_feat, tgt, prot_seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #return self.data[idx]
        #onlypredictのときのみ
        protein_feat, tgt, prot_seq = self.data[idx]          
        return protein_feat, tgt, prot_seq

def custom_collate_fn(batch):
    protein_feats, tgt_seqs, uids = zip(*batch)
    protein_feats = torch.stack(protein_feats)
    tgt_seqs = pad_sequence(tgt_seqs, batch_first=True, padding_value=config.rna_vocab_NAR["<pad>"])
    return protein_feats, tgt_seqs, list(uids)