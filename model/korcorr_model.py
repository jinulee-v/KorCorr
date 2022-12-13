import torch
import torch.nn as nn

from model.transformer import Transformer

torch.set_printoptions(linewidth=210)

class KorCorrModel(Transformer):
    def __init__(self, vocab_size, label_size, hidden_dim, encoder_layers, encoder_heads, encoder_pf_dim, encoder_dropout, decoder_layers, decoder_heads, decoder_pf_dim, decoder_dropout, pad_idx, bos_idx, eos_idx, max_length, use_seqcls_decoding, device):
        super().__init__(vocab_size, hidden_dim, encoder_layers, encoder_heads, encoder_pf_dim, encoder_dropout, decoder_layers, decoder_heads, decoder_pf_dim, decoder_dropout, pad_idx, bos_idx, eos_idx, max_length, device)

        self.seqlabel = nn.Linear(hidden_dim, label_size)
        self.use_seqcls_decoding = use_seqcls_decoding
    
    def forward(self, src, tgt):
        enc_src, output, _ = super().forward(src, tgt)
        
        label_logits = self.seqlabel(enc_src)
        return label_logits, output
    
    def generate(self, src):
        # If not using sequence classification decoding,
        # Apply simple greedy decoding
        if not self.use_seqcls_decoding:
            return super().generate(src)
        
        # Else, apply copy mechanism.
        # if self.use_seqcls_decoding:

        #src = [batch size, src len]
                
        src_mask = self.make_src_mask(src)
        
        #src_mask = [batch size, 1, 1, src len]
        
        enc_src = self.encoder(src, src_mask)
        label_logits = self.seqlabel(enc_src)
        labels = torch.argmax(label_logits, dim=2)

        # labels = [batch_size, seq_len]
        # copy_idx = [batch_size]
        # copy_streak = [batch_size]

        copy_idx = torch.ones(enc_src.size(0), dtype=torch.long, device=self.device)
        copy_streak = torch.ones(enc_src.size(0), dtype=torch.bool, device=self.device)

        # Method for stepwise greedy generation
        src = torch.cat([src, torch.ones(src.size(0), 1, dtype=torch.long, device=self.device)*self.tgt_pad_idx], dim=1) # append single pad
        labels = torch.cat([labels, torch.zeros(src.size(0), 1, dtype=torch.long, device=self.device)*self.tgt_pad_idx], dim=1) # append single pad
        tgt = torch.ones(src.size(0), 1, dtype=torch.long, device=self.device) * self.bos_idx
        # print(src.size())
        # print(src)
        # print(labels)
        for i in range(self.max_length-2):
            # print()
            # print("copy_idx          ", copy_idx.tolist())
            # print("tokens in copy_idx", torch.gather(src, 1, copy_idx.unsqueeze(1)).squeeze(1).tolist())
            # print("labels in copy_idx", torch.gather(labels, 1, copy_idx.unsqueeze(1)).squeeze(1).tolist())
            # print("copy_streak       ", copy_streak.tolist())
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

            # if copy streak is on, try to copy from the input
            # do_copy = [batch_size, 1]
            do_copy = torch.gather(labels, 1, copy_idx.unsqueeze(1)).squeeze(1) != 2 # If labels are 0/1(copy the token)
            do_copy = torch.logical_and(do_copy, copy_streak.to(torch.bool)) # and is currently on the streak
            output = torch.where(do_copy, torch.gather(src, 1, copy_idx.unsqueeze(1)).squeeze(1), output)
            # print("output            ", output.tolist())

            # output: [batch_size, 1]

            # Set output to [PAD] if batch generation is complete
            output = output * (tgt[:, -1] != self.eos_idx)
            output = output * (tgt[:, -1] != self.tgt_pad_idx)
            output += (output == 0) * self.tgt_pad_idx
            
            # Concat output with current generation
            tgt = torch.cat([tgt, output.unsqueeze(1)], dim=1)

            # print()
            # print(tgt)

            if (tgt[:, -1] - self.tgt_pad_idx).count_nonzero() == 0:
                # Generation finished
                return tgt
            
            # Update copy index and copy streak
            prev_copy_streak = torch.clone(copy_streak)
            copy_streak = torch.logical_or(
                torch.logical_and(
                    copy_streak,
                    torch.gather(labels, 1, copy_idx.unsqueeze(1)).squeeze(1) == 0 # If copied a token and its label is 0
                ),
                torch.logical_and(
                    torch.logical_not(copy_streak),
                    output == torch.gather(src, 1, copy_idx.unsqueeze(1)).squeeze(1) # or if generation reaches copy token, assume copy streak has begun
                )
            )
            # shift copy_idx when..
            move = torch.logical_or(do_copy, copy_streak) # performed copy or generated token is equal as copied one
            # do not move if reached end of sequence
            move *= (copy_idx < src.size(1) - 1) # has not reached end of the sequence
            move *= (torch.gather(src, 1, copy_idx.unsqueeze(1)).squeeze(1) != self.tgt_pad_idx) # reached pad
            copy_idx += move
            # Move copy_idx cursor forward until it reaches a token to copy
            while True:
                move = (copy_idx < src.size(1) - 1) # has not reached end of the sequence
                move *= (torch.gather(labels, 1, copy_idx.unsqueeze(1)).squeeze(1) == 2) # targets skip tokens(2)
                move *= (torch.gather(src, 1, copy_idx.unsqueeze(1)).squeeze(1) != self.tgt_pad_idx) # reached pad
                if torch.equal(move, torch.zeros_like(move)):
                    break
                copy_idx += move

        return tgt