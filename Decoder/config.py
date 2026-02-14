import torch

# --- データパス設定 ---
protein_feat_path = "t30_150M_3D.pt"
csv_path = "ppi3d.csv"

# --- RNA辞書設定 ---
#逐次生成
rna_vocab = {"A": 0, "U": 1, "C": 2, "G": 3, "<pad>": 4, "<sos>": 5, "<eos>": 6}
rna_ivocab = {v: k for k, v in rna_vocab.items()}

# --- モデル・学習設定 ---
input_dim = 640
batch_size = 16
epochs = 500
lr = 0.0001
num_layers = 5
max_len = 102
min_len = 10
top_k = 4
num_samples = 1
temp = 1.5
prot_max_len = 1000

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 保存先設定 ---
save_model = "t30_150M_decoder.pt"
save_lossplot = "loss_plot.png"
