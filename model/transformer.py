"""
https://github.com/bentrevett/pytorch-seq2seq
6 - Attention is All You Need
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = (self.tok_embedding(src) * self.scale)
        src += self.pos_embedding(pos)
        src = self.dropout(src)
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        
        #tgt = [batch size, tgt len]
        #enc_src = [batch size, src len, hid dim]
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        
        pos = torch.arange(0, tgt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, tgt len]
        
        tgt = (self.tok_embedding(tgt) * self.scale) + self.pos_embedding(pos)
        tgt = self.dropout(tgt)
                
        #tgt = [batch size, tgt len, hid dim]
        
        for layer in self.layers:
            tgt, attention = layer(tgt, enc_src, tgt_mask, src_mask)
        
        #tgt = [batch size, tgt len, hid dim]
        #attention = [batch size, n heads, tgt len, src len]
        
        output = self.fc_out(tgt)
        
        #output = [batch size, tgt len, output dim]
            
        return output, attention        


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        
        #tgt = [batch size, tgt len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _tgt, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        
        #dropout, residual connection and layer norm
        tgt = self.self_attn_layer_norm(tgt + self.dropout(_tgt))
            
        #tgt = [batch size, tgt len, hid dim]
            
        #encoder attention
        _tgt, attention = self.encoder_attention(tgt, enc_src, enc_src, src_mask)
        #dropout, residual and layer norm
        tgt = self.ff_layer_norm(tgt + self.dropout(_tgt))
        
        #tgt = [batch size, tgt len, hid dim]
        #attention = [batch size, n heads, tgt len, src len]
        
        return tgt, attention
        

class Transformer(nn.Module):
    def __init__(self, 
                 vocab_size,
                 hidden_dim,
                 encoder_layers,
                 encoder_heads,
                 encoder_pf_dim,
                 encoder_dropout, 
                 decoder_layers,
                 decoder_heads,
                 decoder_pf_dim,
                 decoder_dropout,
                 pad_idx,
                 bos_idx,
                 eos_idx,
                 max_length,
                 device):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size,
            hidden_dim,
            encoder_layers,
            encoder_heads,
            encoder_pf_dim,
            encoder_dropout,
            device=device,
            max_length=max_length
        )
        self.decoder = Decoder(
            vocab_size,
            hidden_dim,
            decoder_layers,
            decoder_heads,
            decoder_pf_dim,
            decoder_dropout,
            device=device,
            max_length=max_length
        )
        self.src_pad_idx = pad_idx
        self.tgt_pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_length = max_length
        self.device = device

        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
        self.apply(initialize_weights)
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_tgt_mask(self, tgt):
        
        #tgt = [batch size, tgt len]
        
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #tgt_pad_mask = [batch size, 1, 1, tgt len]
        
        tgt_len = tgt.shape[1]
        
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device = self.device)).bool()
        
        #tgt_sub_mask = [tgt len, tgt len]
            
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        
        return tgt_mask

    def forward(self, src, tgt):
        
        #src = [batch size, src len]
        #tgt = [batch size, tgt len]
                
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        #src_mask = [batch size, 1, 1, src len]
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        
        #output = [batch size, tgt len, output dim]
        #attention = [batch size, n heads, tgt len, src len]
        
        return enc_src, output, attention
    
    def generate(self, src):
        #src = [batch size, src len]
                
        src_mask = self.make_src_mask(src)
        
        #src_mask = [batch size, 1, 1, src len]
        #tgt_mask = [batch size, 1, tgt len, tgt len]
        
        enc_src = self.encoder(src, src_mask)

        # Method for stepwise greedy generation
        def step(tgt, enc_src, src_mask):
            #tgt = [batch size, tgt len]
            #enc_src = [batch size, src len, hid dim]
            #tgt_mask = [batch size, 1, tgt len, tgt len]
            #src_mask = [batch size, 1, 1, src len]
            tgt_mask = self.make_tgt_mask(tgt)
            output, attention = self.decoder.forward(tgt, enc_src, tgt_mask, src_mask)
            #output = [batch size, tgt len, output dim]
            output = output[:, -1, :]
            output = torch.argmax(output, dim=-1)
            # output = [batch_size]

            # Set to [PAD] if batch generation is complete
            output = output * (tgt[:, -1] != self.eos_idx)
            output = output * (tgt[:, -1] != self.tgt_pad_idx)
            output += (output == 0) * self.tgt_pad_idx
            return torch.cat([tgt, output.unsqueeze(1)], dim=1)
        
        tgt = torch.ones(src.size(0), 1, dtype=torch.long).to(self.device) * self.bos_idx
        pads = torch.ones(src.size(0), dtype=torch.long).to(self.device) * self.tgt_pad_idx
        for i in range(self.max_length-2):
            tgt = step(tgt, enc_src, src_mask)
            if (tgt[:, -1] - pads).count_nonzero() == 0:
                # Generation finished
                return tgt
        return tgt