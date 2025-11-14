import torch
import torch.nn as nn
import Decoder.config as config
#import config

def generate_square_subsequent_mask(sz, *, device=None, dtype=None):
    m = torch.full((sz, sz), float('-inf'), device=device, dtype=dtype)
    return torch.triu(m, diagonal=1).masked_fill(torch.eye(sz, device=device, dtype=torch.bool), 0)

class ProteinToRNA(nn.Module):
    def __init__(self, input_dim, num_layers, vocab_size=len(config.rna_vocab), embed_dim=256, nhead=8, max_len=config.max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.rna_vocab["<pad>"])
        self.pos_encoder = nn.Parameter(torch.randn(max_len, embed_dim))
        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.mem_pos = nn.Parameter(torch.randn(max_len, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, protein_feat, tgt_seq):
        B, L = tgt_seq.size(0), tgt_seq.size(1)
        pad_id = self.embedding.padding_idx

        # 追加
        sos_id = config.rna_vocab["<sos>"]
        sos = torch.full((B, 1), sos_id, device=tgt_seq.device, dtype=tgt_seq.dtype)
        tgt_in = torch.cat([sos, tgt_seq[:, :-1]], dim=1)

        # --- RNA側埋め込み ---
        L_in = tgt_in.size(1)                                             
        pos_enc = self.pos_encoder[:L_in].unsqueeze(0)                  
        tgt_emb = self.embedding(tgt_in) + pos_enc
        tgt_mask = generate_square_subsequent_mask(                      
            L_in, device=tgt_seq.device, dtype=tgt_emb.dtype
        )
        tgt_key_padding_mask = (tgt_in == pad_id)            

        # --- タンパク質側メモリ ---
        if protein_feat.dim() == 2:
            # [B, D] → [1, B, E]（グローバルベクトル）
            memory = self.input_proj(protein_feat).unsqueeze(0)          # [1, B, E]
            memory = memory + self.mem_pos[:1].unsqueeze(1)              # 位置0の埋め込みを加算
        elif protein_feat.dim() == 3:
            # [B, S, D] → [S, B, E]（配列として活かす）
            B_, S, D = protein_feat.size()
            assert B_ == B, "batch mismatch"
            mem = self.input_proj(protein_feat)                           # [B, S, E]
            mem = mem.transpose(0, 1)                                     # [S, B, E]
            mem = mem + self.mem_pos[:S].unsqueeze(1)                     # [S, B, E]
            memory = mem
        else:
            raise ValueError(f"protein_feat must be [B, D] or [B, S, D], got {protein_feat.shape}")

        out = self.decoder(
            tgt=tgt_emb.transpose(0, 1),     # [L, B, E]
            memory=memory,                   # [S or 1, B, E]
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(out.transpose(0, 1))

class ProteinToRNA_NAR(nn.Module):
    def __init__(self, input_dim, num_layers,
                 vocab_size=None, embed_dim=256, nhead=8, max_len=config.max_len):
        super().__init__()
        self.vocab_size = vocab_size or len(config.rna_vocab_NAR)
        self.embed_dim  = embed_dim
        self.max_len    = max_len

        self.input_proj = nn.Linear(input_dim, embed_dim)

        pad_id = config.rna_vocab_NAR["<pad>"]
        self.token_embed = nn.Embedding(self.vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_embed   = nn.Parameter(torch.empty(max_len, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.mem_pos = nn.Parameter(torch.empty(max_len, embed_dim))
        nn.init.normal_(self.mem_pos, std=0.02)

        self.mask_id = config.rna_vocab_NAR.get("<MASK>")

        dec_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=False)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.fc_out   = nn.Linear(embed_dim, self.vocab_size)

    def _build_tgt(self, B: int, L: int, device: torch.device) -> torch.Tensor:
        mask_vec = self.token_embed.weight[self.mask_id]
        tgt = mask_vec.view(1, 1, self.embed_dim).expand(L, B, self.embed_dim).to(device)
        pos = self.pos_embed[:L].unsqueeze(1).expand(L, B, self.embed_dim).to(device)
        return tgt + pos

    def _encode_protein(self, protein_feat: torch.Tensor) -> torch.Tensor:
        """
        protein_feat: [B, D] もしくは [B, S, D]
        戻り値: memory [S, B, E]
        """
        if protein_feat.dim() == 2:
            # [B, D] → [1, B, E]
            mem = self.input_proj(protein_feat).unsqueeze(0)  # [1, B, E]
            mem = mem + self.mem_pos[:1].unsqueeze(1)         # [1, B, E]
        elif protein_feat.dim() == 3:
            # [B, S, D] → [S, B, E]
            B, S, D = protein_feat.size()
            proj = self.input_proj(protein_feat.view(B * S, D))   # [B*S, E]
            mem = proj.view(B, S, self.embed_dim).transpose(0, 1) # [S, B, E]
            if S > self.max_len:
                raise ValueError(f"protein length S({S}) exceeds max_len({self.max_len}); increase max_len")
            mem = mem + self.mem_pos[:S].unsqueeze(1)             # [S, B, E]
        else:
            raise ValueError(f"protein_feat must be [B, D] or [B, S, D], got {protein_feat.shape}")
        return mem

    def _decode_logits(self, protein_feat: torch.Tensor, L: int) -> torch.Tensor:
        """MASK入力で一括デコードして logits を返す。"""
        assert L <= self.max_len, f"out_len({L}) > max_len({self.max_len})"
        B, device = protein_feat.size(0), protein_feat.device
        memory = self._encode_protein(protein_feat)        # [S, B, E]
        # tgt: [L, B, E]（因果マスクなし＝全位置同時）
        tgt = self._build_tgt(B, L, device)
        dec_out = self.decoder(tgt, memory)                # [L, B, E]
        return self.fc_out(dec_out.transpose(0, 1))        # [B, L, V]

    def forward(self, protein_feat: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """学習/評価用：tgt_seqの“中身は使わず” 長さ L だけ参照。"""
        L = tgt_seq.size(1)
        return self._decode_logits(protein_feat, L)

    def forward_parallel(self, protein_feat: torch.Tensor, out_len: int) -> torch.Tensor:
        """推論で長さを指定して一括 logits（勾配不要）。"""
        return self._decode_logits(protein_feat, int(out_len))

    @torch.no_grad()
    def generate(self, protein_feat: torch.Tensor, out_len: int) -> torch.Tensor:
        """argmax で一括生成（[B, L]）。"""
        logits = self.forward_parallel(protein_feat, out_len)
        return logits.argmax(dim=-1)
