import torch
import config

def greedy_decode(model, protein_feat, max_len=config.max_len):
    model.eval()
    generated = [config.rna_vocab["<sos>"]]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_seq = torch.tensor(generated, device=protein_feat.device).unsqueeze(0)
            output = model(protein_feat.unsqueeze(0), tgt_seq)
            next_token = output[0, -1].argmax().item()
            if next_token == config.rna_vocab["<eos>"]:
                break
            generated.append(next_token)
    return generated[1:] 

def sample_decode(model, protein_feat, max_len=config.max_len, num_samples=config.num_samples, top_k=config.top_k, temperature=1.0):
    model.eval()
    feat = protein_feat.unsqueeze(0)
    sos_id = config.rna_vocab["<sos>"]
    eos_id = config.rna_vocab["<eos>"]
    results = []

    with torch.no_grad():
        while len(results) < num_samples:
            seq = [sos_id]
            for _ in range(max_len):
                tgt_seq = torch.tensor(seq, device=feat.device).unsqueeze(0)
                output = model(feat, tgt_seq)
                logits = output[0, -1] / temperature  # [V]
                probs = torch.softmax(logits, dim=-1)

                # 上位 top_k の確率とインデックス
                topk_probs, topk_idx = probs.topk(top_k)
                mask = torch.zeros_like(probs)
                mask[topk_idx] = topk_probs
                probs_filtered = mask / mask.sum()

                next_token = torch.multinomial(probs_filtered, 1).item()
                if next_token == eos_id:
                    break
                seq.append(next_token)

            # <sos> を除去して結果に登録
            out_seq = seq[1:]
            if out_seq:
                results.append(out_seq)

    return results