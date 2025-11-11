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

def sample_decode(model, protein_feat, max_len=config.max_len, num_samples=config.num_samples, top_k=config.top_k, temperature=1.0):
    device = protein_feat.device
    '''feat = protein_feat.unsqueeze(0)
    feat = feat.to(device)'''
    single = False
    if protein_feat.dim() == 1:
        feat = protein_feat.unsqueeze(0)
        single = True
    elif protein_feat.dim() == 2:
        feat = protein_feat  

    B = feat.size(0)  
    sos_id = config.rna_vocab["<sos>"]
    eos_id = config.rna_vocab["<eos>"]

    tgt = torch.full((B, 1), sos_id, dtype=torch.long, device=device)  # [B, 1]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        # logits: [B, T, V] → 直近トークン
        logits = model(feat, tgt)[:, -1, :] / temperature  # [B, V]
        probs = torch.softmax(logits, dim=-1)              # [B, V]

        # top-k
        if top_k and top_k > 0 and top_k < probs.size(1):
            topk_probs, topk_idx = probs.topk(top_k, dim=-1)  # [B, k]
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topk_idx, topk_probs)
            probs = mask / (mask.sum(dim=1, keepdim=True) + 1e-12)

        # サンプリング（終了済みには <eos> を流し込む）
        next_tok = torch.multinomial(probs, 1)  # [B,1]
        eos_col = torch.full_like(next_tok, eos_id)
        next_tok = torch.where(finished.unsqueeze(1), eos_col, next_tok)

        tgt = torch.cat([tgt, next_tok], dim=1)   # [B, t+1]
        finished |= (next_tok.squeeze(1) == eos_id)
        if finished.all():
            break

    # <sos> 以降を取り出し、<eos>で打ち切り
    def strip_one(row: torch.Tensor):
        ids = row.tolist()[1:]
        out = []
        for x in ids:
            if x == eos_id:
                break
            out.append(int(x))
        return out

    outs = [strip_one(tgt[b]) for b in range(B)]
    return outs[0] if single else outs

def sample_decode_reinforce(model, protein_feat, max_len=config.max_len, num_samples=1, top_k=config.top_k, temperature=1.0, min_len=config.min_len):
    if protein_feat.dim() == 1:
        feat = protein_feat.unsqueeze(0)
        single = True
    elif protein_feat.dim() == 2:
        feat = protein_feat
        single = False
    else:
        raise ValueError(f"protein_feat.dim() must be 1 or 2, got {protein_feat.dim()}")

    B = feat.size(0)
    dev = feat.device

    sos_id = config.rna_vocab["<sos>"]
    eos_id = config.rna_vocab["<eos>"]
    pad_id = config.rna_vocab["<pad>"]  

    # ---- ① 生成（no_grad）----
    all_seqs = []
    with torch.no_grad():
        for b in range(B):
            feat_b = feat[b:b+1]  # [1, D]
            seqs_b = []
            for _ in range(num_samples):
                seq = []
                prefix = torch.tensor([[sos_id]], device=dev)  # [1,1]
                for t in range(max_len):
                    logits = model(feat_b, prefix)[:, -1, :]  # [1, V]
                    next_logits = logits.clone()

                    # 禁止トークン
                    next_logits[:, pad_id] = float("-inf")
                    next_logits[:, sos_id] = float("-inf")

                    # 最小長と最終ステップの制約
                    if t == max_len - 1:
                        next_logits.fill_(float("-inf"))
                        next_logits[:, eos_id] = 0.0
                    elif t < min_len:
                        next_logits[:, eos_id] = float("-inf")

                    # 温度
                    if temperature is not None and temperature > 0.0 and temperature != 1.0:
                        next_logits = next_logits / temperature

                    if top_k and top_k > 0 and top_k < next_logits.size(1):
                        topv, topi = torch.topk(next_logits, k=top_k, dim=-1)
                        filtered = torch.full_like(next_logits, float("-inf"))
                        filtered.scatter_(1, topi, topv)
                        probs = torch.softmax(filtered, dim=-1)
                    else:
                        probs = torch.softmax(next_logits, dim=-1)

                    next_tok = torch.multinomial(probs, 1).item()
                    if next_tok == eos_id:
                        break
                    seq.append(next_tok)
                    prefix = torch.cat([prefix, torch.tensor([[next_tok]], device=dev)], dim=1)
                seqs_b.append(seq)
            all_seqs.append(seqs_b)

    # ----  logp 再計算（no_gradのまま／学習側で使わないなら省略可）----
    logp_rows = []
    with torch.no_grad():
        for b in range(B):
            feat_b = feat[b:b+1]
            rows_b = []
            for seq in all_seqs[b]:
                if len(seq) == 0:
                    rows_b.append(torch.tensor(0.0, device=dev))
                    continue
                inp = torch.tensor([[sos_id] + seq[:-1]], device=dev)  # [1, L]
                logits = model(feat_b, inp)                             # [1, L, V]
                log_probs = torch.log_softmax(logits[0], dim=-1)        # [L, V]
                target = torch.tensor(seq, device=dev).unsqueeze(1)     # [L,1]
                rows_b.append(log_probs.gather(1, target).sum())
            logp_rows.append(torch.stack(rows_b))

    if single:
        return all_seqs[0], logp_rows[0]  # List[List[int]], Tensor[num_samples]
    else:
        if num_samples == 1:
            return [s[0] for s in all_seqs], torch.stack([r[0] for r in logp_rows])  # List[List[int]], Tensor[B]
        else:
            return all_seqs, torch.stack(logp_rows)

        
def sample_decode_multi(model,
                  protein_feat,
                  max_len=config.max_len,
                  num_samples=config.num_samples,
                  top_k=config.top_k,
                  temperature=1.0):
    
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
                  temperature=1.0):
    
    device = protein_feat.device
    if protein_feat.dim() == 1:
        feat = protein_feat.unsqueeze(0)
        single = True
    elif protein_feat.dim() == 2:
        feat = protein_feat
    else:
        raise ValueError(f"protein_feat.dim() must be 1 or 2, got {protein_feat.dim()}")

    B = feat.size(0)
    out_all = [[] for _ in range(B)]
    seen_all = [set() for _ in range(B)] 

    # ---- 定数（NAR版からNAR_NAR参照を外し、rna_vocabに統一）----
    PAD = config.rna_vocab["<pad>"]
    SOS = config.rna_vocab["<sos>"]
    EOS = config.rna_vocab["<eos>"]

    target = int(num_samples) if num_samples is not None else 1
    S_step = getattr(config, "samples_chunk_size", 0) or min(target, 8)

    rounds = math.ceil(target / S_step)

    for r in range(rounds):
        # このラウンドで各タンパク質から並列生成する本数（S）
        remaining = target - (rounds - 1) * S_step if r == rounds - 1 else S_step
        S = max(1, remaining)  # ← これが “S” です

        # [B*S, D]
        feat_rep = feat.repeat_interleave(S, dim=0).to(device, non_blocking=True)
        N = feat_rep.size(0)

        # <sos> 初期化
        toks = torch.full((N, 1), SOS, dtype=torch.long, device=device)
        finished = torch.zeros(N, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = model(feat_rep, toks)          # [N, L, V]
            next_logits = logits[:, -1, :].clone()

            # 生成禁止: <pad>, <sos>
            next_logits[:, PAD] = float("-inf")
            next_logits[:, SOS] = float("-inf")

            # 最小長未満は <eos> 禁止（短すぎ対策）
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
                continue  # 既に満たしていれば追加しない

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
                    continue  # 短すぎは除外（重複は許可）

                out_all[i].append(seq)

    # ── ここで“絶対”本数保証：不足分は既存配列を繰り返して充当（重複許可前提）──
    for i in range(B):
        n = len(out_all[i])
        if n < target:
            if n == 0:
                print(f"[WARN] 生成配列が0件: idx={i}. 空のリストを繰り返して埋めるのは避けました。"
                      " min_len と max_len / top_k / temperature を確認してください。")
            else:
                k = 0
                while len(out_all[i]) < target:
                    out_all[i].append(out_all[i][k % n])
                    k += 1

    if target == 1:
        return [seqs[0] if len(seqs) > 0 else [] for seqs in out_all]
    return out_all