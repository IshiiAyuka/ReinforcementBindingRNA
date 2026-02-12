import random
import torch
from decode import sample_decode_multi_AR, sample_decode_multi
import config

def _ids_to_string(ids):
    ivocab = config.rna_ivocab
    sos = config.rna_vocab["<sos>"]
    eos = config.rna_vocab["<eos>"]
    pad = config.rna_vocab["<pad>"]

    # tuple返し (ids, score) みたいなのも吸収
    if isinstance(ids, tuple):
        ids = ids[0]

    # TensorならCPUへ、2次元なら先頭系列を取る
    if torch.is_tensor(ids):
        ids = ids.detach().cpu()
        if ids.dim() == 2:
            ids = ids[0]
        ids = ids.tolist()

    # list of list なら先頭系列を取る
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
        ids = ids[0]

    out = []
    for t in ids:
        if torch.is_tensor(t):
            t = int(t.item())
        else:
            t = int(t)

        if t == eos:
            break
        if t in (sos, pad):
            continue
        out.append(ivocab[t])
    return "".join(out)
    
def show_test_samples(model, dataset, device):
    model.eval()
    n_show = 5
    print(f"\n==== Testデータからランダムに{n_show}件表示 ====\n", flush=True)

    # 同じサンプルが重複しないように選ぶ
    indices = random.sample(range(len(dataset)), k=n_show)

    with torch.no_grad():
        for i, sample_idx in enumerate(indices, start=1):
            protein_feat, rna_target, _ = dataset[sample_idx]

            protein_feat = protein_feat.unsqueeze(0).to(device)

            pred_ids = sample_decode_multi_AR(
                model,
                protein_feat,
                num_samples=config.num_samples,
            )  

            predicted_seq = _ids_to_string(pred_ids)

            target_core = rna_target[1:-1] 
            target_seq = "".join(
                [config.rna_ivocab[int(t.item())] for t in target_core]
            )

            print(f"--- サンプル {i} ---", flush=True)
            print("正解testRNA配列:", flush=True)
            print(target_seq, flush=True)
            print("予測testRNA配列:", flush=True)
            print(predicted_seq, flush=True)
            print()
