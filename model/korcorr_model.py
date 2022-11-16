import torch
import torch.nn as nn

from model.transformer import Transformer

class KorCorrModel(Transformer):
    def __init__(self, vocab_size, label_size, hidden_dim, encoder_layers, encoder_heads, encoder_pf_dim, encoder_dropout, decoder_layers, decoder_heads, decoder_pf_dim, decoder_dropout, pad_idx, bos_idx, eos_idx, max_length, device):
        super().__init__(vocab_size, hidden_dim, encoder_layers, encoder_heads, encoder_pf_dim, encoder_dropout, decoder_layers, decoder_heads, decoder_pf_dim, decoder_dropout, pad_idx, bos_idx, eos_idx, max_length, device)

        self.seqlabel = nn.Linear(hidden_dim, label_size)
    
    def forward(self, src, tgt):
        enc_src, output, _ = super().forward(src, tgt)
        
        label_logits = self.seqlabel(enc_src)
        return label_logits, output