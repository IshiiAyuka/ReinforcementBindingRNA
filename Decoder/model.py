import torch
import torch.nn as nn
import config

def generate_square_subsequent_mask(sz, *, device=None, dtype=None):
    m = torch.full((sz, sz), float('-inf'), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1).masked_fill(torch.eye(sz, device=device, dtype=torch.bool), 0)

class ProteinToRNA(nn.Module):
    def __init__(self, input_dim, num_layers, vocab_size=len(config.rna_vocab), embed_dim=256, nhead=8, max_len=config.max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.rna_vocab["<pad>"])
        self.pos_encoder = nn.Parameter(torch.randn(max_len, embed_dim))
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.mem_pos = nn.Parameter(torch.randn(config.prot_max_len, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, protein_feat, tgt_seq):
        B, L = tgt_seq.shape
        pad_id = self.embedding.padding_idx
                                         
        pos_enc = self.pos_encoder[:L].unsqueeze(0)                  
        tgt_emb = self.embedding(tgt_seq) + pos_enc
        tgt_mask = generate_square_subsequent_mask(L, device=tgt_seq.device, dtype=tgt_emb.dtype)
        tgt_key_padding_mask = (tgt_seq == pad_id)   

        B2, S, D = protein_feat.shape
        memory = self.input_proj(protein_feat)
        memory = memory + self.mem_pos[:S].unsqueeze(0)         

        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)
