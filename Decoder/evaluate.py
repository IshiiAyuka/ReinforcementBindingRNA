import torch
from tqdm import tqdm
from decode import greedy_decode
from utils import global_alignment, local_alignment
import config

def evaluate_model(model, dataset, device):
    recalls = []
    global_align_scores = []
    local_align_scores = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="評価中"):
            protein_feat, rna_target = dataset[i]
            predicted_ids = greedy_decode(model, protein_feat.to(device))
            target_ids = [id.item() for id in rna_target[1:-1]]  # <sos>, <eos> 除去

            predicted_seq = "".join([config.rna_ivocab[i] for i in predicted_ids])
            target_seq = "".join([config.rna_ivocab[i] for i in target_ids])

            if len(predicted_seq) == 0 or len(target_seq) == 0:
                continue

            min_len = min(len(predicted_ids), len(target_ids))
            match_count = sum([predicted_ids[j] == target_ids[j] for j in range(min_len)])
            recall = match_count / len(target_ids) if len(target_ids) > 0 else 0.0
            recalls.append(recall)

            global_align_score = global_alignment(predicted_seq, target_seq)
            local_align_score = local_alignment(predicted_seq, target_seq)

            global_align_scores.append(global_align_score)
            local_align_scores.append(local_align_score)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_global_align = sum(global_align_scores) / len(global_align_scores) if global_align_scores else 0.0
    avg_local_align = sum(local_align_scores) / len(local_align_scores) if local_align_scores else 0.0

    print(f"平均配列回収率 (Recall): {avg_recall:.4f}", flush=True)
    print(f"平均グローバルアライメントスコア (Identity): {avg_global_align:.4f}", flush=True)
    print(f"平均ローカルアライメントスコア (Identity): {avg_local_align:.4f}", flush=True)

    return avg_recall, avg_global_align, avg_local_align
