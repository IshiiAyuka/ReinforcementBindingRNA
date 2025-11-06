import torch

# --- データパス設定 ---
protein_feat_path = "t6_8M.pt"
#clstr_path = "clustered40.fasta.clstr"
csv_path = "ppi3d.csv"

# --- RNA辞書設定 ---
rna_vocab = {"A": 0, "U": 1, "C": 2, "G": 3, "<pad>": 4, "<sos>": 5, "<eos>": 6}
rna_ivocab = {v: k for k, v in rna_vocab.items()}  # idから文字への逆引き辞書も作成

# --- モデル・学習設定 ---
input_dim = 320
batch_size = 2
epochs = 300
lr = 0.0001
num_layers = 5
max_len = 2050
top_k = 4
num_samples = 1

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 保存先設定 ---
save_model = "t6_8M_decoder.pt"
save_lossplot = "loss_plot.png"
