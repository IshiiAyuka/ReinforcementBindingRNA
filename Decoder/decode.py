import torch
import torch.nn as nn
#import Decoder.config as config
import config 

MAX_BODY_LEN = config.max_len - 2
if MAX_BODY_LEN <= 0:
    raise ValueError("config.max_len must be at least 2 to allow <sos> and <eos>")

@torch.no_grad()
def greedy_decode(model, protein_feat):

    model.eval()

    eos_id = config.rna_vocab["<eos>"]
    pad_id = config.rna_vocab.get("<pad>", None)

    generated = []

    pf = protein_feat.unsqueeze(0) if protein_feat.dim() == 1 else protein_feat
    pf = pf.to(config.device, non_blocking=True)

    with torch.no_grad():
        for _ in range(MAX_BODY_LEN):
            if len(generated) == 0:
                tgt_seq = torch.empty((1, 0), dtype=torch.long, device=config.device)
            else:
                tgt_seq = torch.tensor(generated, dtype=torch.long, device=config.device).unsqueeze(0)

            output = model(pf, tgt_seq)      # [1, T', V]
            logits = output[0, -1].clone()   # 次トークンのロジット

            cur_len = len(generated)
            if cur_len < config.min_len:
                logits[eos_id] = -1e9
            if pad_id is not None:
                logits[pad_id] = -1e9

            next_token = int(torch.argmax(logits).item())
            if (next_token == eos_id) and (cur_len >= config.min_len):
                break
            generated.append(next_token)

    return generated

def _debug_check_decoding(logits, next_logits, probs, t: int):
    # 1) モデル出力 logits のチェック
    has_nan_logit = torch.isnan(logits).any().item()
    has_inf_logit = torch.isinf(logits).any().item()

    if has_nan_logit or has_inf_logit:
        print("===== [DEBUG] logits has NaN/Inf in sample_decode_multi_AR =====", flush=True)
        print(f"  t = {t}, logits.shape = {logits.shape}", flush=True)
        print(f"  logits.min = {logits.min().item():.4e}, logits.max = {logits.max().item():.4e}", flush=True)
        bad_mask = torch.isnan(logits) | torch.isinf(logits)
        bad_rows = bad_mask.any(dim=-1).any(dim=-1).nonzero(as_tuple=True)[0]
        print(f"  bad row indices = {bad_rows.tolist()}", flush=True)
        i = bad_rows[0].item()
        print("  logits[bad_row][:3, :5] =",
              logits[i, :3, :5].detach().cpu().tolist(), flush=True)
        raise RuntimeError("DEBUG: logits NaN/Inf just after model()")

    # 2) next_logits の有限値チェック（マスク・top_k 後）
    finite_mask = torch.isfinite(next_logits)
    no_finite = ~finite_mask.any(dim=-1)   # その行に有限値が1つもない
    if no_finite.any():
        bad_idx = no_finite.nonzero(as_tuple=True)[0]
        print("===== [DEBUG] next_logits row has no finite value =====", flush=True)
        print(f"  t = {t}, indices = {bad_idx.tolist()}", flush=True)
        j = bad_idx[0].item()
        print("  next_logits[bad_row][:10] =",
              next_logits[j, :10].detach().cpu().tolist(), flush=True)
        print(f"  next_logits.min = {next_logits.min().item():.4e}, "
              f"max = {next_logits.max().item():.4e}", flush=True)
        raise RuntimeError("DEBUG: all -inf/NaN in next_logits row before softmax")

    # 3) probs のチェック（softmax 後）
    has_nan_prob = torch.isnan(probs).any().item()
    has_inf_prob = torch.isinf(probs).any().item()
    has_neg_prob = (probs < 0).any().item()
    row_sum = probs.sum(dim=-1)  # [N]
    bad_sum = (row_sum <= 0) | torch.isnan(row_sum) | torch.isinf(row_sum)

    if has_nan_prob or has_inf_prob or has_neg_prob or bad_sum.any().item():
        print("===== [DEBUG] probs invalid just before multinomial =====", flush=True)
        print(f"  t = {t}, probs.shape = {probs.shape}", flush=True)
        print(f"  probs.min = {probs.min().item():.4e}, probs.max = {probs.max().item():.4e}", flush=True)
        print(f"  has_nan_prob = {has_nan_prob}, has_inf_prob = {has_inf_prob}, has_neg_prob = {has_neg_prob}", flush=True)
        bad_rows = bad_sum.nonzero(as_tuple=True)[0].tolist()
        print(f"  row_sum<=0 or NaN/Inf rows = {bad_rows}", flush=True)
        if bad_rows:
            k = bad_rows[0]
            print("  probs[bad_row][:10] =",
                  probs[k, :10].detach().cpu().tolist(), flush=True)
            print("  row_sum[bad_row] =", row_sum[k].item(), flush=True)
        raise RuntimeError("DEBUG: invalid probs before torch.multinomial")

@torch.inference_mode()
def sample_decode_multi_AR(model,
                  protein_feat,
                  max_len=config.max_len,
                  num_samples=config.num_samples,
                  top_k=config.top_k,
                  temperature=config.temp):
    
    device = next(model.parameters()).device

    protein_feat = protein_feat.to(device, non_blocking=True)
    B, S_seq, D_in = protein_feat.shape
    max_body_len = max_len - 2

    PAD = config.rna_vocab["<pad>"]
    SOS = config.rna_vocab["<sos>"]
    EOS = config.rna_vocab["<eos>"]

    feat_rep = protein_feat.repeat_interleave(num_samples, dim=0)
    N = feat_rep.size(0) 

    # 生成用トークン列
    toks = torch.full((N, 1), SOS, dtype=torch.long, device=device)  # [N, 1]
    finished = torch.zeros(N, dtype=torch.bool, device=device)

    for t in range(max_body_len):
        # logits: [N, L, V]
        logits = model(feat_rep, toks)
        next_logits = logits[:, -1, :].clone()  # [N, V]

        # 生成禁止: <pad>, <sos>
        next_logits[:, PAD] = float("-inf")
        next_logits[:, SOS] = float("-inf")

        # 最小長未満は <eos> 禁止
        cur_len_no_sos = toks.size(1) - 1
        if cur_len_no_sos < config.min_len:
            next_logits[~finished, EOS] = float("-inf")

        # 温度
        if temperature and temperature != 1.0:
            next_logits = next_logits / max(temperature, 1e-6)

        # top-k
        V = next_logits.size(-1)
        if top_k and 0 < top_k < V:
            topv, topi = torch.topk(next_logits, k=top_k, dim=-1)
            masked = torch.full_like(next_logits, float("-inf"))
            next_logits = masked.scatter(1, topi, topv)

        # softmax → サンプリング
        probs = torch.softmax(next_logits, dim=-1)      # [N, V]
        next_ids = torch.multinomial(probs, 1).squeeze(1)  # [N]

        # 終了済み行は EOS 固定
        if finished.any():
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, EOS),
                next_ids,
            )

        toks = torch.cat([toks, next_ids.unsqueeze(1)], dim=1)  # [N, L+1]
        finished |= (next_ids == EOS)

        if finished.all():
            break

    # ========= トークン列 → ID列（タンパク質ごと / サンプルごと） =========
    seqs_all = []  # 長さ N (= B * num_samples) を想定

    for row in toks:  # row: [L]
        seq = []
        for tok in row.tolist():
            if tok == EOS:
                break
            if tok in (PAD, SOS):
                continue
            seq.append(tok)

        if len(seq) < config.min_len:
            continue  # 通常は起こらない想定（EOS 禁止しているため）

        seqs_all.append(seq)

    # ========= B × num_samples の形にグルーピング =========
    out_all = [[] for _ in range(B)]
    expected = B * num_samples
    total = min(len(seqs_all), expected)

    idx = 0
    for i in range(B):
        for _ in range(num_samples):
            if idx >= total:
                break
            out_all[i].append(seqs_all[idx])
            idx += 1

    # ========= 返り値の形（B と num_samples に応じて） =========
    if num_samples == 1:
    # 常に「タンパク質ごとに1本」: List[List[int]] (len = B)
        return [seqs[0] if len(seqs) > 0 else [] for seqs in out_all]

    # 一般ケース: List[List[List[int]]]  (B × num_samples × 可変長)
    return out_all
