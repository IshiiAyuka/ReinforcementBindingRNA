import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import random
from tqdm import tqdm
import time
import gzip
from Bio.PDB.MMCIFParser import MMCIFParser
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

features = torch.load("filtered_protein_features.pt")

# RNA辞書
rna_vocab = {"A": 0, "U": 1, "C": 2, "G": 3, "I": 4, "<pad>": 5, "<sos>": 6, "<eos>": 7}
rna_ivocab = {v: k for k, v in rna_vocab.items()}

def extract_rna_sequence_from_cif(file_path):
    parser = MMCIFParser(QUIET=True)
    rna_seq = ""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as handle:
            structure = parser.get_structure("complex", handle)
            model = next(structure.get_models())
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()
                    if resname in rna_vocab:
                        rna_seq += resname
    except Exception as e:
        print(f"エラー（RNA抽出）: {file_path} - {e}")
    return rna_seq

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

# Dataset定義
class RNADataset(Dataset):
    def __init__(self, protein_feat_file, cif_dir):
        full_feats_dict = torch.load(protein_feat_file)  # 例: {"2zni_A": tensor(...), ...}
        self.cif_dir = cif_dir
        self.data = []

        # チェーンIDを除いてPDB IDでまとめる
        feats_by_pdb = defaultdict(list)
        for k, v in full_feats_dict.items():
            pdb_id = k.split("_")[0]
            feats_by_pdb[pdb_id].append(v)

        for fname in os.listdir(cif_dir):
            if not fname.endswith(".cif.gz"):
                continue
            complex_id = os.path.splitext(os.path.splitext(fname)[0])[0]  # "1abc.cif.gz" → "1abc"
            cif_path = os.path.join(cif_dir, fname)

            if complex_id not in feats_by_pdb:  
                print(f"特徴量が存在しないためスキップ: {complex_id}")
                continue

            rna_seq = extract_rna_sequence_from_cif(cif_path)
            if not rna_seq:
                print(f"RNA配列が存在しないためスキップ: {complex_id}")
                continue

            protein_feats = feats_by_pdb[complex_id]
            protein_feat = torch.stack(protein_feats, dim=0).mean(dim=0)
            try:
                with gzip.open(cif_path, 'rt', encoding='utf-8') as handle:
                    structure = MMCIFParser(QUIET=True).get_structure("complex", handle)
                    model = next(structure.get_models())

                    for chain in model:
                        rna_seq = ""
                        for residue in chain:
                            resname = residue.get_resname()
                            if resname in rna_vocab:
                                rna_seq += resname

                        if len(rna_seq) == 0 or len(rna_seq) > 500:
                            continue  # 長さ制限または空を除外

                        tgt = torch.tensor(
                            [rna_vocab["<sos>"]] + [rna_vocab[c] for c in rna_seq if c in rna_vocab] + [rna_vocab["<eos>"]],
                            dtype=torch.long
                        )
                        self.data.append((protein_feat, tgt))

            except Exception as e:
                print(f"エラー（cif読み込み）: {cif_path} - {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Transformer Decoder
class ProteinToRNA(nn.Module):
    def __init__(self, input_dim, vocab_size, embed_dim=128, num_layers=2, nhead=4, max_len=500):
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
        memory = self.input_proj(protein_feat).unsqueeze(1)  # shape: (B, 1, embed_dim)
        memory = memory.transpose(0, 1)  # shape: (1, B, embed_dim)
        tgt_mask = generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        output = self.decoder(tgt_emb.transpose(0, 1), memory, tgt_mask=tgt_mask)
        return self.fc_out(output.transpose(0, 1))

# Greedy Decode
def greedy_decode(model, protein_feat, max_len=500):
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

# データ読み込み
dataset = RNADataset("filtered_protein_features.pt", "./filtered_data")  # cif_folderにはcif.gzが格納されている前提

train_size = int(0.95 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
print(f"全データ数: {len(dataset)}")
print(f"train データ数: {len(train_dataset)}")

# モデル定義・学習
model = ProteinToRNA(input_dim=320, vocab_size=len(rna_vocab)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=rna_vocab["<pad>"])

# 学習ループ
epochs = 100
loss_history = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    for protein_feat, tgt_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        protein_feat = protein_feat.to(device)
        tgt_seq = tgt_seq.to(device)

        optimizer.zero_grad()
        output = model(protein_feat, tgt_seq[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_seq[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")


# モデル保存
torch.save(model.state_dict(), "decoder_model.pt")
print("Transformer Decoder モデルを保存しました。")

# --- グラフ描画 ---
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()

# --- 複数サンプルの予測表示 ---
num_samples = 5  # 表示したいサンプル数
print(f"\n==== テストデータからランダムに {num_samples} 件表示 ====\n")

for i in range(num_samples):
    sample_idx = random.randint(0, len(test_dataset) - 1)
    protein_feat, rna_target = test_dataset[sample_idx]
    predicted_ids = greedy_decode(model, protein_feat.to(device))
    predicted_seq = "".join([rna_ivocab[i] for i in predicted_ids])
    target_seq = "".join([rna_ivocab[i.item()] for i in rna_target[1:-1]])  # <sos>, <eos>除去

    print(f"--- サンプル {i+1} ---")
    print("正解RNA配列:")
    print(target_seq)
    print("予測RNA配列:")
    print(predicted_seq)
    print()
