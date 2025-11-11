import torch
from tqdm import tqdm
from decode import sample_decode_multi, sample_decode_multi_AR
from utils import global_alignment, local_alignment
import config

def evaluate_model(model, loader, device):
    recalls = []
    global_align_scores = []
    local_align_scores = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="評価中"):
            protein_batch = batch[0]
            rna_tgt_batch = batch[1]

            B = protein_batch.size(0)
            for i in range(B):
                protein_feat = protein_batch[i].to(device)
                predicted_ids = sample_decode_multi_AR(model, protein_feat)     
                target_ids = rna_tgt_batch[i].view(-1).tolist()[1:-1]

                predicted_seq = "".join([config.rna_ivocab_NAR[i] for i in predicted_ids])
                target_seq = "".join([config.rna_ivocab_NAR[i] for i in target_ids])

                if not predicted_ids or not target_ids:
                    continue

                min_len = min(len(predicted_ids), len(target_ids))
                match_count = sum([predicted_ids[j] == target_ids[j] for j in range(min_len)])
                recalls.append(match_count / len(target_ids))

                global_align_scores.append(global_alignment(predicted_seq, target_seq))
                local_align_scores.append(local_alignment(predicted_seq, target_seq))

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_global_align = sum(global_align_scores) / len(global_align_scores) if global_align_scores else 0.0
    avg_local_align = sum(local_align_scores) / len(local_align_scores) if local_align_scores else 0.0

    print(f"平均配列回収率 (Recall): {avg_recall:.4f}", flush=True)
    print(f"平均グローバルアライメントスコア (Identity): {avg_global_align:.4f}", flush=True)
    print(f"平均ローカルアライメントスコア (Identity): {avg_local_align:.4f}", flush=True)

    return avg_recall, avg_global_align, avg_local_align


def evaluate_model_NAR(model, loader, device):
    recalls = []
    global_align_scores = []
    local_align_scores = []

    # NAR用 ivocab があれば優先
    ivocab = getattr(config, "rna_ivocab_NAR", getattr(config, "rna_ivocab"))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="評価中"):
            protein_batch = batch[0]
            rna_tgt_batch = batch[1]

            B = protein_batch.size(0)
            for i in range(B):
                protein_feat = protein_batch[i].to(device)

                # 返り値は [B][L] 相当（B=1なら [[L]]）なので先頭を取る
                pred_ids_list = sample_decode_multi(
                    model,
                    protein_feat,
                    num_samples=config.num_samples,
                    max_len=config.max_len,
                    top_k=config.top_k,
                    temperature=1.0,
                )
                predicted_ids = pred_ids_list[0] if isinstance(pred_ids_list, list) else pred_ids_list  # [L]

                target_ids = rna_tgt_batch[i].view(-1).tolist()[1:-1]  # <sos>, <eos>除去

                predicted_seq = "".join([config.rna_ivocab_NAR[t] for t in predicted_ids])
                target_seq    = "".join([config.rna_ivocab_NAR[t] for t in target_ids])

                if not predicted_ids or not target_ids:
                    continue

                # 位置一致ベースの recall
                min_len = min(len(predicted_ids), len(target_ids))
                match_count = sum(predicted_ids[j] == target_ids[j] for j in range(min_len))
                recalls.append(match_count / len(target_ids))

                # アライメント評価
                global_align_scores.append(global_alignment(predicted_seq, target_seq))
                local_align_scores.append(local_alignment(predicted_seq, target_seq))

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_global_align = sum(global_align_scores) / len(global_align_scores) if global_align_scores else 0.0
    avg_local_align = sum(local_align_scores) / len(local_align_scores) if local_align_scores else 0.0

    print(f"平均配列回収率 (Recall): {avg_recall:.4f}", flush=True)
    print(f"平均グローバルアライメントスコア (Identity): {avg_global_align:.4f}", flush=True)
    print(f"平均ローカルアライメントスコア (Identity): {avg_local_align:.4f}", flush=True)

    return avg_recall, avg_global_align, avg_local_align