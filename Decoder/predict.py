import random
import torch
from decode import greedy_decode, sample_decode_multi
import config
from tqdm import tqdm

def show_test_samples(model, dataset, device):
    model.eval()
    print(f"\n==== Testデータからランダムに5件表示 ====\n", flush=True)
    with torch.no_grad():
        for i in range(5):
            sample_idx = random.randint(0, len(dataset) - 1)
            protein_feat, rna_target, _ = dataset[sample_idx]

            predicted_ids = greedy_decode(model, protein_feat.to(device))
            predicted_seq = "".join([config.rna_ivocab[i] for i in predicted_ids])
            target_seq = "".join([config.rna_ivocab[i.item()] for i in rna_target[1:-1]])  # <sos>, <eos>除去

            print(f"--- サンプル {i+1} ---", flush=True)
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