import torch
import torch.nn as nn
import config

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

class ProteinToRNA(nn.Module):
    def __init__(self, input_dim, num_layers, vocab_size=7, embed_dim=128, nhead=4, max_len=config.max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.rna_vocab["<pad>"])
        self.pos_encoder = nn.Parameter(torch.randn(max_len, embed_dim))
        self.input_proj = nn.Linear(input_dim, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, protein_feat, tgt_seq):
        pos_enc = self.pos_encoder[:tgt_seq.size(1)].unsqueeze(0)
        tgt_emb = self.embedding(tgt_seq) + pos_enc
        memory = self.input_proj(protein_feat).unsqueeze(1)
        memory = memory.transpose(0, 1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)
        output = self.decoder(tgt_emb.transpose(0, 1), memory, tgt_mask=tgt_mask)
        return self.fc_out(output.transpose(0, 1))
