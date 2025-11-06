import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from Decoder.model import ProteinToRNA
import Decoder.config as config
from Decoder.dataset import FastaProteinDataset, collate_rl
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Optional
from collections import OrderedDict
from LucaOneTasks.src.predict_v1 import run as predict_run
from Decoder.decode import sample_decode_reinforce

# ================== 入出力 ==================
fasta_path = "swissprot_RBP.fasta"
weights = "/home/slab/ishiiayuka/M2/Decoder/t30_150M_decoder_AR_1003.pt"
protein_feat_path = "t30_150M_swissprot_RBP.pt"
output_path = "/home/slab/ishiiayuka/M2/Decoder/t30_150M_decoder_AR_after_reinforce_swissprot"

# ================== ヘルパ関数 ==================
def logprob_from_logits(logits, tokens, pad_id, eos_id):
    # logits: [B,L,V], tokens: [B,L] → logp: [B]
    logp_tok = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B,L]
    not_pad = (tokens != pad_id)
    eos_mask = (tokens == eos_id)
    csum = eos_mask.int().cumsum(dim=1)
    before_eos = (csum == 0)                 # eosより前（eos自身は含まず）
    first_eos  = eos_mask & (csum == 1)      # 最初のeosのみ
    include = (before_eos | first_eos) & not_pad
    return (logp_tok * include.float()).sum(dim=-1)

def tokens_to_strings(tokens, ivocab, eos_id, pad_id, sos_id):
    # tokens: [B,L] → list[str]（<eos>で打ち切り、<pad>/<sos>は除去）
    seqs = []
    for row in tokens.tolist():
        s = []
        for t in row:
            if t == eos_id:
                break
            if t == pad_id or t == sos_id:
                continue
            s.append(ivocab[int(t)])
        seqs.append("".join(s))
    return seqs

# ================== GPU設定 ==================
device_ids = [0, 1, 2, 3]
device = f'cuda:{device_ids[0]}'
torch.cuda.set_device(device_ids[0])

# ================== ハイパーパラメータ ==================
baseline_mean   =  torch.tensor(0.0, device=device)          # 報酬の移動平均 (= ベースライン)
baseline_alpha  = 0.7
max_steps       = 15
grad_clip_norm  = 0.7
batch_size = 16
MIN_LEN = 10

PRINT_SAMPLES_EVERY_STEP = 1
NUM_SHOW_PER_BATCH      = 5
SEQ_PREVIEW_LEN         = 120
REWARD_GPU_ID = device_ids[0] 

# --- データ準備 ---
dataset_train = FastaProteinDataset(fasta_path, protein_feat_path)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_rl, pin_memory=True, num_workers=4, persistent_workers=True)
torch.manual_seed(42)

# --- モデル定義 ---
model = ProteinToRNA(input_dim=config.input_dim, num_layers=config.num_layers)
state_dict = torch.load(weights, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)

model.to(device)
model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# --- 報酬関数定義 ---
class LucaPPIReward:
    def __init__(self, model_path: str = "LucaOneTasks/", topk: Optional[int] = None):
        self.model_path = model_path
        self.topk = topk
        self.common_kwargs = dict(
            model_path = self.model_path,
            llm_truncation_seq_length = 500,
            dataset_name = "ncRPI",
            dataset_type = "gene_protein",
            task_type = "binary_class",
            task_level_type = "seq_level",
            model_type = "lucappi2",
            input_type = "matrix",
            input_mode = "pair",
            threshold = 0.5,
            step = "716380",
            time_str = "20240404105148",
            topk = topk,
            gpu_id = REWARD_GPU_ID,
            emb_dir = None,
            matrix_embedding_exists = False
        )
    def __call__(self, prot_list: list[str], rna_list: list[str]) -> torch.Tensor:
        assert len(prot_list) == len(rna_list)
        sequences = [["", "", "gene", "prot", rna_list[i], prot_list[i], ""] for i in range(len(prot_list))]
        with torch.no_grad():
            results = predict_run(sequences, **self.common_kwargs)
        probs = [row[4] for row in results]
        return torch.tensor(probs, dtype=torch.float32, device=device)

luca_reward = LucaPPIReward(model_path="LucaOneTasks/", topk=None)

# --- ヘルパ: トークン列 -> 文字列 ---
def pack_tokens_from_seqs(seqs, eos_id, pad_id, L):
    B = len(seqs)
    out = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        s = s[: max(0, L-1)]
        row = s + [eos_id]
        out[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return out

# --------------------------- 学習ループ --------------------------- #
eos_id = config.rna_vocab["<eos>"]
pad_id = config.rna_vocab["<pad>"]
sos_id = config.rna_vocab["<sos>"]

for step in tqdm(range(max_steps), desc="step", position=0):
    model.train()

    optimizer.zero_grad(set_to_none=True)
    accum_steps = len(train_loader)

    for batch_idx, (protein_feat, tgt_seqs, protein_seq_list) in enumerate(tqdm(train_loader, desc="Samples", position=1, leave=False)):
        protein_feat = protein_feat.to(device, non_blocking=True)

        # ========= 1) 逐次生成 =========
        with torch.no_grad():
            rna_idlists, _ = sample_decode_reinforce(
            model, protein_feat,
            max_len=config.max_len,
            num_samples=1,
            top_k=config.top_k,
            temperature=1.0,
            min_len=config.min_len
        )
        tokens = pack_tokens_from_seqs(rna_idlists, eos_id, pad_id, L=config.max_len)

        # ========= 3) 報酬 =========
        rna_strs = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)
        with torch.no_grad():
            R = luca_reward(protein_seq_list, rna_strs)
            if R.dim() == 0:
                R = R.expand(tokens.size(0))

        # ========= 4) logp =========  
        prefix = torch.full((tokens.size(0), 1), sos_id, device=device, dtype=torch.long)
        L_TF = min(tokens.size(1), config.max_len - 1)
        inp  = torch.cat([prefix, tokens[:, :L_TF]], dim=1)   # [B, 1+L_TF]
        logits = model(protein_feat, inp)[:, 1:, :] 
        tokens_tf = tokens[:, :logits.size(1)]                # [B, L_TF]

        logp_batch = logprob_from_logits(logits, tokens_tf, pad_id=pad_id, eos_id=eos_id)


        # ========= 5) REINFORCE損失 =========
        with torch.no_grad():
            R_mean = R.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean

        advantage = (R - baseline_mean).detach()
        loss = -(advantage * logp_batch).mean()

        (loss / accum_steps).backward()

        # ログ出力
        if (step % PRINT_SAMPLES_EVERY_STEP == 0):
            R_cpu = R.detach().float().cpu().tolist()
            for i, (seq, r) in enumerate(zip(rna_strs[:NUM_SHOW_PER_BATCH], R_cpu[:NUM_SHOW_PER_BATCH])):
                preview = seq[:SEQ_PREVIEW_LEN]
                tail = "..." if len(seq) > SEQ_PREVIEW_LEN else ""
                print(f"[gen] step={step:03d} batch={batch_idx:04d} idx={i:02d} len={len(seq):4d} R={r:.4f}  {preview}{tail}", flush=True)

    clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    save_path = f"{output_path}_{step+1}step.pt"  
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_path)
