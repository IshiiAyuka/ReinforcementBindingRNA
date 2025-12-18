import random
from collections import OrderedDict

import subprocess
import re

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from Decoder.model import ProteinToRNA
import Decoder.config as config
from Decoder.dataset import ProteinFeatFastaDictDataset, custom_collate_fn_feat_fasta_dict

from torch.utils.data import DataLoader
from tqdm import tqdm

from Decoder.decode import sample_decode_multi_AR
from LucaOneTasks.src.predict_v1 import run as predict_run
import multiprocessing as mp


# ===========================
#  報酬: マルチGPU並列プール
# ===========================
def _reward_worker(task_q, result_q, gpu_id: int, common_kwargs: dict):
    while True:
        item = task_q.get()
        if item is None:
            break
        job_id, indices, prot_list, rna_list = item
        sequences = [["", "", "rna", "prot", rna_list[i], prot_list[i], ""] for i in range(len(prot_list))]

        kwargs = dict(common_kwargs)
        kwargs["gpu_id"] = gpu_id  # 強制上書き

        results = predict_run(sequences, **kwargs)
        probs = [row[4] for row in results]  # luca の5列目を確率と解釈

        result_q.put((job_id, indices, probs))


class LucaPPIRewardPool:
    def __init__(self, gpu_ids, common_kwargs, max_workers=None):
        self.ctx = mp.get_context("spawn")
        self.gpu_ids = list(gpu_ids) if gpu_ids else [0]
        if max_workers is None:
            max_workers = len(self.gpu_ids)

        self.task_queues = []
        self.procs = []
        self.result_q = self.ctx.Queue()

        for i in range(max_workers):
            tq = self.ctx.Queue(maxsize=2)
            self.task_queues.append(tq)
            p = self.ctx.Process(
                target=_reward_worker,
                args=(tq, self.result_q, self.gpu_ids[i % len(self.gpu_ids)], common_kwargs),
                daemon=True
            )
            p.start()
            self.procs.append(p)

        self._next_job_id = 0

    def score_pairs(self, prot_list, rna_list, device: str = "cpu"):
        N = len(prot_list)

        num_workers = len(self.procs)
        chunks = []
        for w in range(num_workers):
            start = (N * w) // num_workers
            end = (N * (w + 1)) // num_workers
            if start >= end:
                continue
            indices = list(range(start, end))
            p_chunk = prot_list[start:end]
            r_chunk = rna_list[start:end]
            job_id = self._next_job_id
            self._next_job_id += 1
            self.task_queues[w].put((job_id, indices, p_chunk, r_chunk))
            chunks.append(job_id)

        out = [None] * N
        for _ in range(len(chunks)):
            job_id, indices, probs = self.result_q.get()
            for idx, val in zip(indices, probs):
                out[idx] = val

        return torch.tensor(out, dtype=torch.float32, device=device)

    def close(self):
        for q in self.task_queues:
            q.put(None)
        for p in self.procs:
            p.join(timeout=5)


# ===========================
#  既存ヘルパ
# ===========================
def logprob_from_logits(logits, tokens, pad_id, eos_id):
    logp_tok = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # [B,L]
    not_pad = (tokens != pad_id)
    eos_mask = (tokens == eos_id)
    csum = eos_mask.int().cumsum(dim=1)
    before_eos = (csum == 0)
    first_eos = eos_mask & (csum == 1)
    include = (before_eos | first_eos) & not_pad
    num = (logp_tok * include.float()).sum(dim=-1)
    den = include.float().sum(dim=-1).clamp_min(1.0)
    return num / den  # 長さ正規化（平均）


def tokens_to_strings(tokens, ivocab, eos_id, pad_id, sos_id):
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


# ===========================
#  RNAfold
# ===========================
def compute_rnafold_energies(rna_strs, device, energy_type: str = "ensemble"):
    """
    energy_type: "ensemble" または "mfe"
    """

    energies = [0.0] * len(rna_strs)

    non_empty = [(i, s) for i, s in enumerate(rna_strs) if s]
    if not non_empty:
        return torch.tensor(energies, dtype=torch.float32, device=device)

    seqs = [s for _, s in non_empty]
    proc = subprocess.run(
        ["RNAfold", "-p", "--noPS"],
        input="\n".join(seqs) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )

    # stdoutとstderrをまとめてパースする（環境によって出力先が異なるため）
    out_text = proc.stdout
    if proc.stderr:
        out_text += "\n" + proc.stderr

    lines = out_text.strip().splitlines()

    # アンサンブル自由エネルギーを出現順に収集
    ensemble_vals = []
    mfe_vals = []  # フォールバック用（最小自由エネルギー）
    for ln in lines:
        m_ensemble = re.search(r"free energy of ensemble\s*=\s*([-+]?\d+(?:\.\d+)?)", ln)
        if m_ensemble:
            ensemble_vals.append(float(m_ensemble.group(1)))
        m_mfe = re.search(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*\)", ln)
        if m_mfe:
            mfe_vals.append(float(m_mfe.group(1)))

    # 見つかったものを順番に割り当て、足りなければMFEを代用
    for rank, (idx, _) in enumerate(non_empty):
        if energy_type == "mfe":
            # MFE優先、なければアンサンブルをフォールバック
            if rank < len(mfe_vals):
                energies[idx] = mfe_vals[rank]
            elif rank < len(ensemble_vals):
                energies[idx] = ensemble_vals[rank]
        else:
            # アンサンブル優先、なければMFEをフォールバック
            if rank < len(ensemble_vals):
                energies[idx] = ensemble_vals[rank]
            elif rank < len(mfe_vals):
                energies[idx] = mfe_vals[rank]

    return torch.tensor(energies, dtype=torch.float32, device=device)


def compute_gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    seq = seq.upper()
    bases = [c for c in seq if c in ("A", "U", "G", "C")]
    if not bases:
        return 0.0
    gc_count = sum(1 for c in bases if c in ("G", "C"))
    return gc_count / len(bases)


def compute_extra_reward_on_target(
    rna_list,
    device,
    *,
    rnafold_target=0.5,
    rnafold_sigma=0.15,
    rnafold_energy_type="ensemble",
    gc_target=0.5,
    gc_sigma=0.15,
    length_target=30,
    length_sigma=15,
    weights=None,
):
    """
    追加報酬をRNAfold・GC含量・長さの3項すべてで計算して合算する。
    weightsで各項の寄与を調整可能（デフォルトは全項1.0）。
    """
    weights = weights or {}
    w_rnafold = float(weights.get("rnafold", 1.0))
    w_gc = float(weights.get("gc", 1.0))
    w_length = float(weights.get("length", 1.0))

    k = len(rna_list)
    lengths = torch.tensor([len(s) for s in rna_list], dtype=torch.float32, device=device)  # [k]
    lengths_safe = lengths.clamp_min(1.0)  # RNAfold正規化用

    # RNAfold
    E = compute_rnafold_energies(rna_list, device=device, energy_type=rnafold_energy_type)  # [k]
    rnafold_norm = (-E) / lengths_safe  # [k]
    rnafold_score = torch.exp(-0.5 * ((rnafold_norm - float(rnafold_target)) / float(rnafold_sigma)) ** 2)  # [k]

    # GC
    gc_list = [compute_gc_fraction(s) for s in rna_list]
    gc_t = torch.tensor(gc_list, dtype=torch.float32, device=device)  # [k]
    gc_score = torch.exp(-0.5 * ((gc_t - float(gc_target)) / float(gc_sigma)) ** 2)  # [k]

    # 長さ
    length_score = torch.exp(-0.5 * ((lengths - float(length_target)) / float(length_sigma)) ** 2)  # [k]

    total = (w_rnafold * rnafold_score) + (w_gc * gc_score) + (w_length * length_score)  # [k]
    info = dict(
        rnafold_score=rnafold_score,
        rnafold_norm=rnafold_norm,
        gc_score=gc_score,
        gc_fraction=gc_t,
        length_score=length_score,
        length_raw=lengths,
    )
    return total, info


# ===========================
#  メイン
# ===========================
def main():
    weights = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_1129.pt"
    protein_feat_path = "/home/slab/ishiiayuka/M2/t30_150M_swissprot_RBP_3D.pt"
    output_path = "/home/slab/ishiiayuka/M2/Decoder/weights/t30_150M_decoder_AR_reinforce_1215_8.pt"

    fasta_path = "/home/slab/ishiiayuka/M2/swissprot_RBP.fasta"

    # --- GPU割り当て ---
    device_ids = [2]
    all_gpus = list(range(torch.cuda.device_count()))
    reward_gpu_ids = [g for g in all_gpus if g not in device_ids]

    device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device_ids[0])

    # --- ハイパーパラメータ ---
    baseline_mean = torch.tensor(0.0, device=device)
    baseline_alpha = 0.5
    max_steps = 10000
    grad_clip_norm = 0.7
    batch_size = 16
    entropy_bonus = 0.01
    seed = 42

    OFFTARGET_LAMBDA = 1

    # オンターゲット/オフターゲットのサンプリング数
    reward_k = 2   # バッチからオンターゲットに使う個数
    neg_m = 5      # 各オンターゲットに対するオフターゲット数

    # 追加報酬（RNAfold / GC / Length）：全項を合算
    RNAFOLD_LAMBDA = 0.7
    GC_LAMBDA = 0.7
    LENGTH_LAMBDA = 0.7
    RNAFOLD_ENERGY_TYPE = "mfe"  # "ensemble" or "mfe"
    RNAFOLD_TARGET = 0.25
    RNAFOLD_SIGMA = 0.1
    GC_TARGET = 0.55
    GC_SIGMA = 0.075
    LENGTH_TARGET = 70
    LENGTH_SIGMA = 30

    random.seed(seed)
    torch.manual_seed(seed)

    # --- データ準備 ---
    dataset_train = ProteinFeatFastaDictDataset(protein_feat_path, fasta_path)
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn_feat_fasta_dict(batch, dataset_train.feat_dict),
        pin_memory=True,
        num_workers=0,
    )
    train_iter = iter(train_loader)

    # --- モデル定義 ---
    model = ProteinToRNA(
        input_dim=config.input_dim,
        num_layers=config.num_layers,
        vocab_size=len(config.rna_vocab),
        max_len=config.max_len
    )
    state_dict = torch.load(weights, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # --- LucaOne 共通引数（プールへ渡す） ---
    common_kwargs = dict(
        model_path="LucaOneTasks/",
        llm_truncation_seq_length=100,
        dataset_name="ncRPI",
        dataset_type="gene_protein",
        task_type="binary_class",
        task_level_type="seq_level",
        model_type="lucappi2",
        input_type="matrix",
        input_mode="pair",
        threshold=0.5,
        step="716380",
        time_str="20240404105148",
        topk=None,
        gpu_id=reward_gpu_ids[0] if len(reward_gpu_ids) > 0 else 0,
        emb_dir=None,
        matrix_embedding_exists=False
    )

    # --- 報酬プール初期化 ---
    luca_pool = LucaPPIRewardPool(gpu_ids=reward_gpu_ids, common_kwargs=common_kwargs)

    eos_id = config.rna_vocab["<eos>"]
    pad_id = config.rna_vocab["<pad>"]
    sos_id = config.rna_vocab["<sos>"]

    for step in tqdm(range(max_steps), desc="Steps"):
        try:
            protein_feat, _, protein_seq_list = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            protein_feat, _, protein_seq_list = next(train_iter)

        protein_feat = protein_feat.to(device, non_blocking=True)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        B_batch = len(protein_seq_list)
        k_eff = min(int(reward_k), B_batch)
        if k_eff <= 0:
            continue

        sel_t_full = torch.randperm(B_batch, device=device)[:k_eff]  # Tensor[k_eff]（元バッチindex）
        sel = sel_t_full.tolist()
        prot_sel = [protein_seq_list[i] for i in sel]
        protein_feat_sel = protein_feat.index_select(0, sel_t_full)  # [k_eff, ...] 生成対象だけ

        # ========= 生成（オンターゲット k_eff 本だけ） =========
        model.eval()
        sampled = sample_decode_multi_AR(
            model,
            protein_feat_sel,
            max_len=config.max_len,
            num_samples=1,
            top_k=config.top_k,
            temperature=config.temp,
        )
        model.train()

        B = protein_feat_sel.size(0)  # == k_eff
        L = config.max_len
        tokens = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(sampled):
            ln = min(len(seq), L)
            if ln > 0:
                tokens[i, :ln] = torch.as_tensor(seq[:ln], dtype=torch.long, device=device)
            if ln < L and ln >= config.min_len:
                tokens[i, ln] = eos_id

        # logits用の入力
        sos_col = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        tgt_in = torch.cat([sos_col, tokens[:, :-1]], dim=1)

        # logits（オンターゲット k_eff 本だけ）
        model.eval()
        logits = model(protein_feat_sel, tgt_in)
        model.train()

        # マスキング
        logits = logits.clone()
        logits[..., pad_id] = -1e9
        logits[..., sos_id] = -1e9
        _min_len = min(config.min_len, logits.size(1))
        logits[:, :_min_len, eos_id] = -1e9
        if config.temp != 1.0:
            logits = logits / max(config.temp, 1e-6)

        # 文字列化（オンターゲット k_eff 本だけ）
        rna_sel = tokens_to_strings(tokens, config.rna_ivocab, eos_id, pad_id, sos_id)

        with torch.no_grad():
            R_sel = luca_pool.score_pairs(prot_sel, rna_sel, device=device)  # [k]

        # ========= オフターゲット：各iについて「自分以外」からランダムに m 個→平均 =========
        if B_batch > 1 and neg_m > 0:
            m_eff = min(int(neg_m), B_batch - 1)
            if m_eff <= 0:
                R_off_sel = torch.zeros_like(R_sel)
            else:
                prot_off, rna_off = [], []
                for pos, i in enumerate(sel):
                    candidates = [j for j in range(B_batch) if j != i]
                    neg_js = random.sample(candidates, k=m_eff)  # 重複なし
                    for j in neg_js:
                        prot_off.append(protein_seq_list[j])
                        rna_off.append(rna_sel[pos])

                with torch.no_grad():
                    scores_off = luca_pool.score_pairs(prot_off, rna_off, device=device)  # [k*m_eff]

                R_off_sel = scores_off.view(k_eff, m_eff).mean(dim=1)  # [k]
        else:
            R_off_sel = torch.zeros_like(R_sel)

        # LucaOne 有効報酬（k個）
        R_eff_sel = R_sel - OFFTARGET_LAMBDA * R_off_sel  # [k]

        # ========= 追加報酬（RNAfold / GC / Length：オンターゲット rna_sel のみ） =========
        with torch.no_grad():
            score, score_info = compute_extra_reward_on_target(
                rna_sel,
                device=device,
                rnafold_target=RNAFOLD_TARGET,
                rnafold_sigma=RNAFOLD_SIGMA,
                rnafold_energy_type=RNAFOLD_ENERGY_TYPE,
                gc_target=GC_TARGET,
                gc_sigma=GC_SIGMA,
                length_target=LENGTH_TARGET,
                length_sigma=LENGTH_SIGMA,
                weights=dict(
                    rnafold=RNAFOLD_LAMBDA,
                    gc=GC_LAMBDA,
                    length=LENGTH_LAMBDA,
                ),
            )

        # 最終報酬（k個）
        R_total_sel = R_eff_sel + score  # [k]

        # logp（全体→sel）
        logp = logprob_from_logits(logits, tokens, pad_id=pad_id, eos_id=eos_id)  # [B]

        # エントロピー（全体→sel）
        logp_all = logits.log_softmax(-1)  # [B, L, V]
        probs = logp_all.exp()
        tok_entropy = -(probs * logp_all).sum(dim=-1)  # [B, L]

        not_pad = (tokens != pad_id)
        eos_mask = (tokens == eos_id)
        csum = eos_mask.int().cumsum(dim=1)
        before_eos = (csum == 0)
        first_eos = eos_mask & (csum == 1)
        include = (before_eos | first_eos) & not_pad

        include_f = include.float()
        den_ent = include_f.sum(dim=-1).clamp_min(1.0)
        entropy_batch = (tok_entropy * include_f).sum(dim=-1) / den_ent  # [B]
        entropy_mean = entropy_batch.mean()

        # REINFORCE損失（k個）
        with torch.no_grad():
            R_mean = R_total_sel.mean()
            baseline_mean = baseline_alpha * baseline_mean + (1 - baseline_alpha) * R_mean

        advantage = (R_total_sel - baseline_mean).detach()
        loss = -(advantage * logp).mean()
        loss = loss - entropy_bonus * entropy_mean

        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # ログ
        loss_val = float(loss.detach().cpu())
        R_mean_val = float(R_sel.mean().detach().cpu())
        Roff_mean_val = float(R_off_sel.mean().detach().cpu())
        Reff_mean_val = float(R_eff_sel.mean().detach().cpu())
        score_mean_val = float(score.mean().detach().cpu())
        score_comp_mean_val = {
            "rnafold": float(score_info["rnafold_score"].mean().detach().cpu()),
            "gc": float(score_info["gc_score"].mean().detach().cpu()),
            "length": float(score_info["length_score"].mean().detach().cpu()),
        }
        score_raw_mean_val = {
            "rnafold_norm": float(score_info["rnafold_norm"].mean().detach().cpu()),
            "gc_fraction": float(score_info["gc_fraction"].mean().detach().cpu()),
            "length": float(score_info["length_raw"].mean().detach().cpu()),
        }
        base_val = float(baseline_mean.detach().cpu())

        print(
            f"[step] step={step:06d} loss={loss_val:.5f} "
            f"R={R_mean_val:.5f} Roff={Roff_mean_val:.5f} Reff={Reff_mean_val:.5f} "
            f"score={score_mean_val:.5f} "
            f"score_comp=[rnafold={score_comp_mean_val['rnafold']:.5f} gc={score_comp_mean_val['gc']:.5f} len={score_comp_mean_val['length']:.5f}] "
            f"score_raw=[rnafold_norm={score_raw_mean_val['rnafold_norm']:.5f} gc_frac={score_raw_mean_val['gc_fraction']:.5f} len={score_raw_mean_val['length']:.5f}] "
            f"baseline={base_val:.5f} "
            f"RNAs={'|'.join(rna_sel)}",
            flush=True,
        )

    torch.save(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        output_path
    )
    print(f"[save] Final weights saved to: {output_path}", flush=True)

    luca_pool.close()


if __name__ == "__main__":
    main()
