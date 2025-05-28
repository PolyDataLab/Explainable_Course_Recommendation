import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import data_helpers_v3_cr_genrec_ser as dh

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
        # self.encode = torch.nn.Embedding(num_embeddings=num_items,
        #                                  embedding_dim=self.embedding_dim,
        #                                  padding_idx=0)
        #self.encode = self.item_embeddings
        #self.encode = torch.nn.Parameter(self.item_embeddings, requires_grad=True).to(device)  # False if I do not want to update item embeddings
        # self.config.embedding_dim = self.item_embeddings.size(1)
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
        # Embedding Inputs
        # item_seq = torch.clamp(item_seq, min=0, max=num_items - 1)
        # basket_seq = torch.clamp(basket_seq, min=0, max=num_items - 1)
        # item_seq = torch.clamp(item_seq, min=0, max=self.item_embedding.num_embeddings - 1)
        # basket_seq = torch.clamp(basket_seq, min=0, max=self.item_embedding.num_embeddings - 1)
        # for (i, user) in enumerate(basket_seq):  # shape of x: [batch_size, seq_len, indices of product]
        #     #embed_baskets = torch.Tensor(self.config.seq_len, self.config.embedding_dim, device=self.device)
        #     embed_baskets = torch.zeros(self.config.seq_len, self.embedding_dim, device=self.device)
        #     for (j, basket) in enumerate(user):  # shape of user: [seq_len, indices of product]
        #         #basket = torch.LongTensor(basket).resize_(1, len(basket))
        #         # basket = torch.tensor(basket, device = self.device).unsqueeze(0)
        #         basket = torch.tensor(basket, device = self.device).unsqueeze(0)
        # print("Max index in basket_seq:", basket_seq.max().item())
        # print("Min index in basket_seq:", basket_seq.min().item())
        # print("Embedding num_embeddings:", self.basket_embedding.num_embeddings)
        # item_embeddings = self.item_embedding(basket_seq)
        # position_embeddings = self.position_embedding(position_ids)  # [max_seq_len, max_basket_size, embedding_dim]
        # position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # #basket_ids = torch.arange(self.max_seq_len)
        # basket_embeddings = self.basket_embedding(basket_ids)  # [max_seq_len, embedding_dim]
        # basket_embeddings = basket_embeddings.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_positions, -1)
        #item_emb = self.item_embedding(basket_seq) + self.position_embedding(item_seq)
        #input_embeddings = item_embeddings + position_embeddings + basket_embeddings
        #item_embeddings = self.item_embedding(basket_seq)
        #position_embeddings = self.position_embedding(position_ids)  # [max_seq_len, max_basket_size, embedding_dim]
        #position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
        #basket_ids = torch.arange(self.max_seq_len)
        #basket_embeddings = self.basket_embedding(basket_ids)  # [max_seq_len, embedding_dim]
        #basket_embeddings = basket_embeddings.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_positions, -1)
        #item_emb = self.item_embedding(basket_seq) + self.position_embedding(item_seq)
        # input_embeddings = item_embeddings + position_embeddings + basket_embeddings
        # print("shape of input: ",input_embeddings.shape)
        # #input_emb_3d = input_embeddings.squeeze(-2)  # [batch_size, max_basket_len, embedding_dim]
        # reshaped_input_tensor = input_embeddings.view(input_embeddings.size(0), -1, input_embeddings.size(-1))

        # #basket_emb = self.basket_embedding(basket_seq)

        # # Encoding
        # item_encoded = self.item_encoder(reshaped_input_tensor)
        # #basket_emb = basket_emb.mean(dim=2) 
        # #basket_encoded = self.basket_encoder(basket_emb)

        # # Decoding
        # tgt_mask = self.generate_square_subsequent_mask(target_seq.size(1)).to(self.device)
        # # target_emb = self.item_embedding(target_seq) + self.position_embedding(target_seq)
        # target_emb = self.item_embedding(target_seq)
        # #decoded = self.decoder(target_emb, item_encoded, tgt_mask=tgt_mask)
        # # Decoding
        # decoded = self.decoder(
        #     target_emb.transpose(0, 1),  # Transform to shape [target_seq_len, batch_size, embedding_dim]
        #     item_encoded.transpose(0, 1),  # Transform to shape [source_seq_len, batch_size, embedding_dim]
        #     tgt_mask=tgt_mask  # Shape [target_seq_len, target_seq_len]
        # )
        # Output Predictions
        # output = self.output_layer(decoded)
        #logits = self.output_layer(decoded)  # Shape: 
        #logits = logits.transpose(0, 1)  # Reshape 
        # print(logits.shape)
        # #return output
        # return logits
        # Basket Encoding
        # Encode the baskets
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


        # Optionally, pool the sequence output to get a fixed-size representation
        # dynamic_user = torch.mean(ub_seqs, dim=1)  # Shape: [batch_size, embedding_dim]

        # return dynamic_user
        # Remove padded elements from the output
        # trimmed_output = []
        # for i, length in enumerate(lengths):
        #     trimmed_output.append(attn_output[i, :length])  # Select valid time steps for each sequence

        # # Combine trimmed outputs into a single tensor
        # dynamic_user = torch.cat(trimmed_output, dim=0)  # Shape: [sum(true_lens), embedding_dim]
        # return dynamic_user
        # Remove padded elements from the output
        # trimmed_output = torch.zeros(self.config.batch_size, self.config.seq_len, self.config.embedding_dim)  # Shape: [batch_size, max_true_len, embedding_dim]
        # for i, length in enumerate(lengths):
        #     trimmed_output[i, :length] = attn_output[i, :length]
        # # Create a list of tensors, each with actual_length
        # trimmed_output = []
        # for i, length in enumerate(lengths):
        #     trimmed_output.append(attn_output[i, :length])  # Extract valid time steps for each sequence

        # return trimmed_output  # List 
        

    def generate_square_subsequent_mask(self, sz):
        """Generate a mask for the decoder to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# Hyperparameters and initialization
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_items = 3881
# embedding_dim = 64
# num_heads = 4
# num_layers = 1
# max_seq_len = 20

# model = GenRec(num_items, embedding_dim, num_heads, num_layers, max_seq_len, device).to(device)
# model_save_dir = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/saved_GenRec_model/'
# final_save_path = os.path.join(model_save_dir, "model_final_GenRec_v0.pth")
# #torch.save(model.state_dict(), final_save_path)
# # print(f"Final model saved at {final_save_path}")
# torch.save(model, final_save_path)
# print(f"Final model saved at {final_save_path}")

