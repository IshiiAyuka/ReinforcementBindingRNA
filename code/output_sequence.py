import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import random
from torch.nn.utils.rnn import pad_sequence

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RNA辞書
rna_vocab = {"A": 0, "U": 1, "C": 2, "G": 3, "<pad>": 4, "<sos>": 5, "<eos>": 6}
rna_ivocab = {v: k for k, v in rna_vocab.items()}

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

# Dataset定義
class RNADataset(Dataset):
    def __init__(self, protein_feat_file, csv_path):
        full_feats_dict = torch.load(protein_feat_file)
        self.data = []
        df = pd.read_csv(csv_path, low_memory=False)

        for idx, row in df.iterrows():
            try:
                chain_id = str(row.iloc[9]).strip() 
                rna_seq = str(row.iloc[25]).strip().upper()
                if (len(rna_seq) <= 10) or (len(rna_seq) >= 1000):
                    continue
                complex_key = f"{chain_id}"
                if complex_key not in full_feats_dict:
                    continue
                protein_feat = full_feats_dict[complex_key]
                if any(c not in rna_vocab for c in rna_seq):
                    continue
                tgt = torch.tensor(
                    [rna_vocab["<sos>"]] + [rna_vocab[c] for c in rna_seq] + [rna_vocab["<eos>"]],
                    dtype=torch.long
                )
                self.data.append((protein_feat, tgt))
            except:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# モデル定義
class ProteinToRNA(nn.Module):
    def __init__(self, input_dim, vocab_size, embed_dim=128, num_layers=3, nhead=4, max_len=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=rna_vocab["<pad>"])
        self.pos_encoder = nn.Parameter(torch.randn(max_len, embed_dim))  
        self.input_proj = nn.Linear(input_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, protein_feat, tgt_seq):
        pos_enc = self.pos_encoder[:tgt_seq.size(1)].unsqueeze(0)
        tgt_emb = self.embedding(tgt_seq) + pos_enc
        memory = self.input_proj(protein_feat).unsqueeze(1).transpose(0, 1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        output = self.decoder(tgt_emb.transpose(0, 1), memory, tgt_mask=tgt_mask)
        return self.fc_out(output.transpose(0, 1))

# greedyデコード
def greedy_decode(model, protein_feat, max_len=1000):
    model.eval()
    generated = [rna_vocab["<sos>"]]
    with torch.no_grad():
        for _ in range(max_len):
            tgt_seq = torch.tensor(generated).unsqueeze(0).to(protein_feat.device)
            output = model(protein_feat.unsqueeze(0), tgt_seq)
            next_token = output[0, -1].argmax().item()
            if next_token == rna_vocab["<eos>"]:
                break
            generated.append(next_token)
    return generated[1:]

# --- モデル読み込み ---
model = ProteinToRNA(input_dim=320, vocab_size=len(rna_vocab)).to(device)
model.load_state_dict(torch.load("decoder_model.pt", map_location=device))
model.eval()
print("モデルの重みを読み込みました。")

# --- データ準備 ---
dataset = RNADataset("ppi3d_protein_features.pt", "ppi3d.csv")
train_size = int(0.95 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

# --- テストデータから5件表示 ---
print("\n==== テストデータから5件表示 ====\n")
for i in range(5):
    protein_feat, rna_target = random.choice(test_dataset)
    predicted_ids = greedy_decode(model, protein_feat.to(device))
    predicted_seq = "".join([rna_ivocab[i] for i in predicted_ids])
    target_seq = "".join([rna_ivocab[i.item()] for i in rna_target[1:-1]])
    print(f"[テスト {i+1}]")
    print("正解:", target_seq)
    print("予測:", predicted_seq)
    print()

# --- 訓練データから5件表示 ---
print("\n==== 訓練データから5件表示 ====\n")
for i in range(5):
    protein_feat, rna_target = random.choice(train_dataset)
    predicted_ids = greedy_decode(model, protein_feat.to(device))
    predicted_seq = "".join([rna_ivocab[i] for i in predicted_ids])
    target_seq = "".join([rna_ivocab[i.item()] for i in rna_target[1:-1]])
    print(f"[訓練 {i+1}]")
    print("正解:", target_seq)
    print("予測:", predicted_seq)
    print()
