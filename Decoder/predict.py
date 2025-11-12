import random
import torch
from decode import sample_decode_multi_AR, sample_decode_multi
import config

def _ids_to_string(ids):
    """<sos>/<eos>/<pad> を除いた可読配列に整形。EOS で打ち切る。"""
    ivocab = config.rna_ivocab
    sos = config.rna_vocab["<sos>"]
    eos = config.rna_vocab["<eos>"]
    pad = config.rna_vocab["<pad>"]
    out = []
    for t in ids:
        if t == eos:
            break
        if t in (sos, pad):
            continue
        out.append(ivocab[t])
    return "".join(out)

def show_test_samples(model, dataset, device):
    model.eval()
    n_show = min(5, len(dataset))
    print(f"\n==== Testデータからランダムに{n_show}件表示 ====\n", flush=True)

    # 同じサンプルが重複しないように選ぶ
    indices = random.sample(range(len(dataset)), k=n_show)

    with torch.no_grad():
        for i, sample_idx in enumerate(indices, start=1):
            protein_feat, rna_target, _ = dataset[sample_idx]

            # 生成（10塩基までは EOS 禁止）
            pred_ids = sample_decode_multi_AR(model,protein_feat)

            if isinstance(pred_ids[0], (list, tuple)):
                pred_ids = pred_ids[0]

            # 整形：表示は A/U/G/C... のみを連結（EOSで打ち切り）
            predicted_seq = _ids_to_string(pred_ids)

            # 正解側は <sos>/<eos> を落としてから表示
            target_core = rna_target[1:-1]  # <sos>, <eos> 除去
            target_seq = "".join([config.rna_ivocab[int(t.item())] for t in target_core])

            print(f"--- サンプル {i} ---", flush=True)
            print("正解testRNA配列:", flush=True)
            print(target_seq, flush=True)
            print("予測testRNA配列:", flush=True)
            print(predicted_seq, flush=True)
            print()

def show_test_samples_NAR(model, dataset, device):
    model.eval()
    print(f"\n==== Testデータからランダムに5件表示 ====\n", flush=True)

    with torch.no_grad():
        for i in range(5):
            sample_idx = random.randint(0, len(dataset) - 1)
            protein_feat, rna_target, _ = dataset[sample_idx]
            protein_feat = protein_feat.to(device)

            # --- 非逐次・一斉デコード（1サンプルだけ） ---
            # 戻り値は [B][L] 相当なので B=1 の先頭を取る
            pred_ids_batch = sample_decode_multi(
                model,
                protein_feat,
                num_samples=config.num_samples,
                max_len=config.max_len,
                top_k=config.top_k,
                temperature=1.0,
            )
            predicted_ids = pred_ids_batch[0]  # [L]

            predicted_seq = "".join([config.rna_ivocab_NAR[i] for i in predicted_ids])
            target_seq = "".join([config.rna_ivocab_NAR[i.item()] for i in rna_target[1:-1]])  # <sos>, <eos>除去

            print(f"--- サンプル {i+1} ---", flush=True)
            print("正解testRNA配列:", flush=True)
            print(target_seq, flush=True)
            print("予測testRNA配列:", flush=True)
            print(predicted_seq, flush=True)
            print()