import torch
import torch.nn as nn
#import Decoder.config as config
import config 
import math

@torch.no_grad()
def greedy_decode(model, protein_feat):
    """
    逐次生成（greedy）。min_len_no_eos までは EOS を強制的に無効化。
    戻り値は <sos> を除いたトークン列。
    """
    model.eval()

    eos_id = config.rna_vocab["<eos>"]
    pad_id = config.rna_vocab.get("<pad>", None)

    generated = []

    # protein_feat が 1D の場合のみバッチ次元を付与（元コードの挙動を踏襲）
    pf = protein_feat.unsqueeze(0) if protein_feat.dim() == 1 else protein_feat
    pf = pf.to(config.device, non_blocking=True)

    with torch.no_grad():
        for _ in range(config.max_len):
            # 接頭辞が空でもOK：空テンソルを渡す（forwardで<sos>付与→長さ1）
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
        
def sample_decode_multi(model,
                  protein_feat,
                  max_len=config.max_len,
                  num_samples=config.num_samples,
                  top_k=config.top_k,
                  temperature=config.temp):
    
    device = protein_feat.device

    single = False
    if protein_feat.dim() == 1:
        feat = protein_feat.unsqueeze(0)
        single = True
    elif protein_feat.dim() == 2:
        feat = protein_feat
    else:
        raise ValueError(f"protein_feat.dim() must be 1 or 2, got {protein_feat.dim()}")

    B = feat.size(0)
    SOS = config.rna_vocab_NAR["<sos>"]
    EOS = config.rna_vocab_NAR["<eos>"]
    PAD = config.rna_vocab_NAR["<pad>"]
    MASK = config.rna_vocab_NAR["<MASK>"]

    # 出力入れ物
    out_all = [[] for _ in range(B)]

    # まとめて複数サンプル（OOMなら後でchunk分割してもOK）
    S_total = int(num_samples) if num_samples is not None else 1
    S_step = getattr(config, "samples_chunk_size", 0) or S_total
    done = 0

    while done < S_total:
        S = min(S_step, S_total - done)

        # [B*S, D] に複製
        feat_rep = feat.repeat_interleave(S, dim=0).to(device, non_blocking=True)

        # ---- NARで一括ロジット ----
        if isinstance(model, nn.DataParallel):
            logits = model.module.forward_parallel(feat_rep, out_len=max_len)  # [B*S, L, V]
        else:
            logits = model.forward_parallel(feat_rep, out_len=max_len)         # [B*S, L, V]

        # 生成したくないトークンをban（<eos>は従来通り許可）
        logits = logits.clone()
        logits[..., PAD]  = float("-inf")
        logits[..., SOS]  = float("-inf")
        logits[..., MASK] = float("-inf")
        logits[:, :min(config.min_len, logits.size(1)), EOS] = float("-inf")

        # 温度
        if temperature and temperature != 1.0:
            logits = logits / max(temperature, 1e-6)

        # top-k（位置ごとにマスク）
        N, L, V = logits.shape  # N = B*S
        flat = logits.view(N * L, V)
        if top_k and 0 < top_k < V:
            topv, topi = torch.topk(flat, k=top_k, dim=-1)
            mask = torch.full_like(flat, float("-inf"))
            mask.scatter_(1, topi, topv)
            flat = mask

        # サンプリング（位置独立）
        probs = torch.softmax(flat, dim=-1)
        toks  = torch.multinomial(probs, 1).view(N, L)  # [B*S, L]

        # 各BについてS本ずつ取り出し、<pad>/<sos>/<MASK>スキップ & <eos>までで切る
        for i in range(B):
            block = toks[i*S:(i+1)*S]
            seq_list = []
            for row in block:
                seq = []
                for tok in row.tolist():
                    if tok == EOS: break
                    if tok in (PAD, SOS, MASK): continue
                    seq.append(tok)
                seq_list.append(seq)
            out_all[i].extend(seq_list)

        done += S

    # 互換：num_samples==1 → [B][L] で返す
    if S_total == 1:
        return [seqs[0] if len(seqs) > 0 else [] for seqs in out_all]
    return out_all

@torch.inference_mode()
def sample_decode_multi_AR(model,
                  protein_feat,
                  max_len=config.max_len,
                  num_samples=config.num_samples,
                  top_k=config.top_k,
                  temperature=config.temp):
    
    device = next(model.parameters()).device

    if protein_feat.dim() == 1:
        feat = protein_feat.unsqueeze(0)
        single = True
    elif protein_feat.dim() == 2:
        feat = protein_feat
    else:
        raise ValueError(f"protein_feat.dim() must be 1 or 2, got {protein_feat.dim()}")

    feat = feat.to(device, non_blocking=True)

    B = feat.size(0)
    out_all = [[] for _ in range(B)]

    PAD = config.rna_vocab["<pad>"]
    SOS = config.rna_vocab["<sos>"]
    EOS = config.rna_vocab["<eos>"]

    target = int(num_samples) if num_samples is not None else 1
    S_step = getattr(config, "samples_chunk_size", 0) or min(target, 8)

    rounds = math.ceil(target / S_step)

    for r in range(rounds):
        remaining = target - (rounds - 1) * S_step if r == rounds - 1 else S_step
        S = max(1, remaining)

        # [B*S, D] 
        feat_rep = feat.repeat_interleave(S, dim=0)
        N = feat_rep.size(0)

        toks = torch.full((N, 1), SOS, dtype=torch.long, device=device)
        finished = torch.zeros(N, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = model(feat_rep, toks)          # [N, L, V]
            next_logits = logits[:, -1, :].clone()

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

            probs = torch.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, 1).squeeze(1)

            # 終了済み行は以後EOS固定
            if finished.any():
                next_ids = torch.where(finished, torch.full_like(next_ids, EOS), next_ids)
            toks = torch.cat([toks, next_ids.unsqueeze(1)], dim=1)
            finished |= (next_ids == EOS)
            if finished.all():
                break

        # BごとにS本へ分割し、<eos>で切り、<pad>/<sos>除外（重複OK）
        for i in range(B):
            if len(out_all[i]) >= target:
                continue

            block = toks[i*S:(i+1)*S]  # [S, L]
            for row in block:
                if len(out_all[i]) >= target:
                    break
                seq = []
                for tok in row.tolist():
                    if tok == EOS: break
                    if tok in (PAD, SOS): continue
                    seq.append(tok)

                if len(seq) < config.min_len:
                    continue

                out_all[i].append(seq)

    # 本数保証
    for i in range(B):
        n = len(out_all[i])
        if n < target:
            if n == 0:
                print(f"[WARN] 生成配列が0件: idx={i}. "
                      "min_len と max_len / top_k / temperature を確認してください。")
            else:
                k = 0
                while len(out_all[i]) < target:
                    out_all[i].append(out_all[i][k % n])
                    k += 1

    if target == 1:
        if B == 1:
            # 単一タンパク質 + 1サンプル → list[int] を返す
            return out_all[0][0] if len(out_all[0]) > 0 else []
        else:
            # バッチ入力のときは従来どおり、各タンパク質ごとに1本ずつ返す
            return [seqs[0] if len(seqs) > 0 else [] for seqs in out_all]
    return out_all
