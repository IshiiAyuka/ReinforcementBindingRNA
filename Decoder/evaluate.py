import torch
from tqdm import tqdm
from decode import sample_decode_multi_AR
from utils import global_alignment, local_alignment
import config

def evaluate_model(model, loader, device):
    recalls = []
    global_align_scores = []
    local_align_scores = []

    model.eval()
    with torch.no_grad():
        for protein_batch, rna_tgt_batch, _ in tqdm(loader, desc="評価中"):
            protein_batch = protein_batch.to(device)
            B = protein_batch.size(0)
            pred_ids_batch = sample_decode_multi_AR(
                model,
                protein_batch,   # [B, S_max, D]
                num_samples=config.num_samples,
            )

            for i in range(B):
                predicted_ids = pred_ids_batch[i]  # List[int]
                target_ids = rna_tgt_batch[i].view(-1).tolist()[1:-1]  # <sos>,<eos>除去

                if not predicted_ids or not target_ids:
                    continue

                # ID → 塩基列
                predicted_seq = "".join([config.rna_ivocab[int(t)] for t in predicted_ids])
                target_seq    = "".join([config.rna_ivocab[int(t)] for t in target_ids])

                # ---- Recall ----
                min_len = min(len(predicted_ids), len(target_ids))
                match_count = sum(predicted_ids[j] == target_ids[j] for j in range(min_len))
                recalls.append(match_count / len(target_ids))

                # ---- アライメント ----
                global_align_scores.append(global_alignment(predicted_seq, target_seq))
                local_align_scores.append(local_alignment(predicted_seq, target_seq))

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_global_align = (sum(global_align_scores) / len(global_align_scores)
                        if global_align_scores else 0.0)
    avg_local_align = (sum(local_align_scores) / len(local_align_scores)
                       if local_align_scores else 0.0)

    print(f"平均配列回収率 (Recall): {avg_recall:.4f}", flush=True)
    print(f"平均グローバルアライメントスコア (Identity): {avg_global_align:.4f}", flush=True)
    print(f"平均ローカルアライメントスコア (Identity): {avg_local_align:.4f}", flush=True)

    return avg_recall, avg_global_align, avg_local_align
