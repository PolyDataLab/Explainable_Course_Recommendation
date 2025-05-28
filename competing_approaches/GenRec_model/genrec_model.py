import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import data_helpers_genrec as dh

class GenRec(nn.Module):
    def __init__(self, config, num_items, num_positions, embedding_dim, n_attn_heads, n_attn_lays, max_seq_len, attn_drops, device): # Config(), num_items, max_basket_size, emb_dim, n_heads, n_attn_lays, max_seq_len, device
        super(GenRec, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_len
        self.num_positions = num_positions
        self.hidden_dim = 128
        self.config = config
        self.n_attn_lays = n_attn_lays
        self.config.embedding_dim = self.embedding_dim
        self.pool = {'avg': dh.pool_avg, 'max': dh.pool_max}[config.basket_pool_type]  # Pooling of basket

        # Embedding Layers
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        # self.position_embedding = nn.Embedding(max_seq_len, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(num_positions, embedding_dim, padding_idx=0)
        self.basket_embedding = nn.Embedding(max_seq_len, embedding_dim, padding_idx=0)

        # Encoders
        self.item_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_attn_heads, dropout= attn_drops),
            num_layers=n_attn_lays
        )
        self.basket_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_attn_heads, dropout= attn_drops),
            num_layers=n_attn_lays
        )

        # Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=n_attn_heads, dropout= attn_drops)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_attn_lays)

        # Output Layer
        self.output_layer = nn.Linear(embedding_dim, max_seq_len)

        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.config.embedding_dim, 
                                  num_heads=n_attn_heads, 
                                  dropout=attn_drops, 
                                  batch_first=True)
            for _ in range(self.n_attn_lays)
        ])

        # Feed-forward layers (optional)
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.embedding_dim)
            )
            for _ in range(self.n_attn_lays)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.config.embedding_dim) for _ in range(self.n_attn_lays)
        ])

    def forward(self, basket_seq, position_ids, basket_ids, batch_size, lengths, target_seq=None): # basket_seq_padded, position_ids, basket_ids, target_seq_padded
       # Prepare basket embeddings
        batch_size, max_seq_len, max_basket_size = basket_seq.size()  # [batch_size, max_seq_len, basket_size]
        ub_seqs = torch.zeros(batch_size, max_seq_len, self.embedding_dim, device=self.device)

        for i, user_baskets in enumerate(basket_seq):
            user_embedded = torch.zeros(max_seq_len, self.embedding_dim, device=self.device)
            for j, basket in enumerate(user_baskets):
                basket_items = basket[basket > 0]  # Filter out padding
                if basket_items.size(0) > 0:
                    # Embed items and add positional embeddings
                    item_embs = self.item_embedding(basket_items)
                    position_ids = torch.arange(basket_items.size(0), device=self.device)
                    position_embs = self.position_embedding(position_ids)
                    item_embs_with_positions = item_embs + position_embs

                    # Pool the basket embeddings
                    pooled_emb = self.pool(item_embs_with_positions, dim=0)
                    user_embedded[j] = pooled_emb
            ub_seqs[i] = user_embedded

        # Add basket position embeddings
        basket_embs = self.basket_embedding(basket_ids)
        basket_embedded = ub_seqs + basket_embs

        # Encoder: Encode basket embeddings
        # Encoder: Encode basket embeddings
        lengths = torch.tensor(lengths, device=self.device)
        mask = torch.arange(max_seq_len, device=self.device).unsqueeze(0).expand(batch_size, max_seq_len) >= lengths.unsqueeze(1)
        basket_encoded = self.basket_encoder(basket_embedded.transpose(0, 1), src_key_padding_mask=mask)
        # Shape: [max_seq_len, batch_size, embedding_dim]
        basket_encoded = basket_encoded.transpose(0, 1)  # [batch_size, max_seq_len, embedding_dim]
        #print("Shape of basket_encoded:", basket_encoded.shape)

        # Decoder: Decode with target sequence (if provided)
        if target_seq is not None:
            # Truncate or pad target sequences to match basket sequence lengths
            max_target_len = lengths.max().item()  # Get the maximum actual length
            if target_seq.size(1) > max_target_len:
                target_seq = target_seq[:, :max_target_len]  # Truncate target sequence
            elif target_seq.size(1) < max_target_len:
                padding = torch.zeros(batch_size, max_target_len - target_seq.size(1), device=self.device, dtype=torch.long)
                target_seq = torch.cat([target_seq, padding], dim=1)  # Pad target sequence

            tgt_emb = self.item_embedding(target_seq)  # Embed target items
            tgt_mask = self.generate_square_subsequent_mask(max_target_len).to(self.device)
            decoded = self.decoder(
                tgt_emb.transpose(0, 1),  # Transform to [tgt_seq_len, batch_size, embedding_dim]
                basket_encoded.transpose(0, 1),  # Transform to [src_seq_len, batch_size, embedding_dim]
                tgt_mask=tgt_mask
            )
            decoded = decoded.transpose(0, 1)  # Shape: [batch_size, tgt_seq_len, embedding_dim]

            # Trim outputs to match actual lengths
            trimmed_outputs = [
                decoded[i, :lengths[i]] for i in range(batch_size)
            ]  # List of tensors with shape [actual_length, embedding_dim]
            #print("output length of decoder: ", trimmed_outputs[0].shape)

            return trimmed_outputs  # List of tensors with actual lengths

        # Return encoder outputs with actual lengths if no target_seq is provided
        trimmed_encoded = [
            basket_encoded[i, :lengths[i]] for i in range(batch_size)
        ]  # List of tensors with shape [actual_length, embedding_dim]
        #print("output length of encoder: ", trimmed_encoded[0].shape)
        return trimmed_encoded
        

    def generate_square_subsequent_mask(self, sz):
        """Generate a mask for the decoder to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


