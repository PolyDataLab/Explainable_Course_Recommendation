#sophos server
# adding one user at a time
import time
import pandas as pd
from preprocess_v1 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
#from topic_model_v2 import *
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import Linear, ModuleList
from torch_geometric.utils import softmax
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch_geometric.utils import degree, dropout_adj
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GATConv

def compute_tfidf_pmi(documents, window_size=10):
    """
    Computes the adjacency matrix A combining TF-IDF and PMI.
    
    Parameters:
    - documents: list of str, corpus of documents.
    - window_size: int, context window size for PMI calculation.
    
    Returns:
    - adjacency_matrix: scipy.sparse.csr_matrix, combined adjacency matrix.
    """
    # Step 1: Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)  # Shape: (num_docs, vocab_size)
    vocab = vectorizer.get_feature_names_out()
    vocab_size = len(vocab)
    vocab_dict_idx_to_wrd = {i: word for i, word in enumerate(vocab)}
    vocab_dict_wrd_to_idx = {word: i for i, word in enumerate(vocab)}
    #print(vocab_size)
    
    # Step 2: PMI Calculation
    # Build a word co-occurrence matrix
    word_cooccurrence = np.zeros((vocab_size, vocab_size))
    
    for doc in documents:
        tokens = doc.split()
        for i, token in enumerate(tokens):
            if token not in vocab:
                continue
            token_idx = np.where(vocab == token)[0][0]
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(tokens))):
                if i == j or tokens[j] not in vocab:
                    continue
                context_idx = np.where(vocab == tokens[j])[0][0]
                word_cooccurrence[token_idx, context_idx] += 1
    
    # Normalize co-occurrence counts to probabilities
    word_sums = word_cooccurrence.sum(axis=1)
    total_sum = word_cooccurrence.sum()
    pmi_matrix = np.zeros_like(word_cooccurrence)
    
    for i in range(vocab_size):
        for j in range(vocab_size):
            if word_cooccurrence[i, j] > 0:
                p_ij = word_cooccurrence[i, j] / total_sum
                p_i = word_sums[i] / total_sum
                p_j = word_sums[j] / total_sum
                pmi_matrix[i, j] = max(0, np.log(p_ij / (p_i * p_j)))  # PMI formula
    
    tfidf_csr = csr_matrix(tfidf_matrix)

    # Create an adjacency matrix of the required shape
    adjacency_matrix = csr_matrix((tfidf_matrix.shape[0]+vocab_size, vocab_size))

    # Fill in document-word (TF-IDF)
    adjacency_matrix[:tfidf_matrix.shape[0], :] = tfidf_csr  # Documents to Words (TF-IDF)

    # Fill in word-word (PMI)
    # adjacency_matrix[:vocab_size, :] = csr_matrix(pmi_matrix)  # Words to Words (PMI)
    adjacency_matrix[tfidf_matrix.shape[0]:, :] = csr_matrix(pmi_matrix)  # Words to Words (PMI)

    
    return adjacency_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx

def get_adj_matrix_text_v2(data, reversed_item_dict_one_hot):

    concept_f_all = {}
    for idx1, cid in enumerate(data["course_code"]):
        #index = data[data['userID'] == user].index.values[0]
        idx2 = reversed_item_dict_one_hot[cid] # cid to idx
        # cid2 = data["course_code"][idx2]
        cname2 = data["course_name"][idx2]
        cid2 = data["course_code"][idx2]
        #text_f = data["course_description"][idx2]
        concept_f = data["concepts"][idx2]
        #row1 = [cname2, text_f]
        # row1 = [cname2, concept_f]
        row1 = [cid2, concept_f]
        concept_f_all[idx2] = row1
        
    concept_f_all_sorted = dict(sorted(concept_f_all.items(), key=lambda item: item[0], reverse=False))
    concept_f_all_new = list(concept_f_all_sorted.values())
    # item_descriptions = data['course_description'].dropna().tolist()
   # item_ids = data['course_code'].dropna().tolist()
    # item_names = data['course_name'].dropna().tolist()
    # item_descriptions = []
    item_concepts = {}
    #item_names = []
    item_id = 0
    for list1 in concept_f_all_new:
        cid3, cconcept = list1
        item_concepts[cid3] = cconcept
        item_id += 1
        #item_names.append(cname3)
    # Get a sorted list of all unique concepts
    #all_concepts = sorted(set.union(*item_concepts.values()))
    all_concepts = sorted(set.union(*map(set, item_concepts.values())))

    num_items = len(item_concepts)
    num_concepts = len(all_concepts)
    vocab_size = num_concepts

    # Create an adjacency matrix (items x concepts)
    adj_matrix = np.zeros((num_items, num_concepts), dtype=int)

    # Fill the adjacency matrix
    concept_to_idx = {concept: idx for idx, concept in enumerate(all_concepts)}
    idx_to_concept = {idx: concept for idx, concept in enumerate(all_concepts)}

    for item, concepts in item_concepts.items():
        for concept in concepts:
            adj_matrix[reversed_item_dict_one_hot[item], concept_to_idx[concept]] = 1  # Mark connection

    print("Binary Adjacency Matrix (Items x Concepts):")
    print(adj_matrix)

    return adj_matrix, vocab_size, concept_to_idx, idx_to_concept

def get_adj_matrix_text_v3(data, reversed_item_dict_one_hot, window_s):

    concept_f_all = {}
    for idx1, cid in enumerate(data["course_code"]):
        #index = data[data['userID'] == user].index.values[0]
        idx2 = reversed_item_dict_one_hot[cid] # cid to idx
        # cid2 = data["course_code"][idx2]
        cname2 = data["course_name"][idx2]
        cid2 = data["course_code"][idx2]
        #text_f = data["course_description"][idx2]
        concept_f = data["concepts"][idx2]
    
        row1 = [cid2, concept_f]
        concept_f_all[idx2] = row1
        
    concept_f_all_sorted = dict(sorted(concept_f_all.items(), key=lambda item: item[0], reverse=False))
    concept_f_all_new = list(concept_f_all_sorted.values())
    
    for list1 in concept_f_all_new:
        cid3, cconcept = list1
        cdesc = ' '.join(cconcept)
        item_descriptions.append(cdesc)
        #item_names.append(cname3)
   
    adj_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx = compute_tfidf_pmi(item_descriptions, window_size=window_s)
   
    #adj_mat, vocab_size = compute_tfidf_pmi(item_concepts, window_size=window_s)
    return adj_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx

def convert_to_one_hot_encoding_cat(data):
    # One-hot encode the 'interactions' column
    data['cat_f'] = data['cat_f'].apply(lambda x: [x])
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(data['cat_f'])
    # Convert back to a DataFrame for easier readability
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_, index=data['item']).reset_index(drop=True)
    cat_dict_one_hot = {index: cat_f for index, cat_f in enumerate(one_hot_df.columns)} # idx , cat
    item_dict_one_hot_cat = {index: item for index, item in enumerate(data['item'])} # idx , cid
    one_hot_df['item'] = data['item']
    list1 =  list(cat_dict_one_hot.values())
    one_hot_df = one_hot_df[['item'] + list1]
    #print(one_hot_df.shape)
    
    return one_hot_encoded, one_hot_df, item_dict_one_hot_cat, cat_dict_one_hot

def convert_to_one_hot_encoding_level(data):
   # One-hot encode the 'interactions' column
    data['level_f'] = data['level_f'].apply(lambda x: [x])
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(data['level_f'])
    # Convert back to a DataFrame for easier readability
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_, index=data['item']).reset_index(drop=True)
    level_dict_one_hot = {index: level_f for index, level_f in enumerate(one_hot_df.columns)} # idx , level
    item_dict_one_hot_level = {index: item for index, item in enumerate(data['item'])} # idx , cid
    one_hot_df['item'] = data['item']
    list1 =  list(level_dict_one_hot.values())
    one_hot_df = one_hot_df[['item'] + list1]
    #print(one_hot_df.shape)
    
    return one_hot_encoded, one_hot_df, item_dict_one_hot_level, level_dict_one_hot

def convert_side_info_to_one_hot_encoding(data, reversed_item_dict_one_hot, num_items):
    # df["category"] 
    # df["level"] 
    cat_f_all = {}
    level_f_all = {}
    for idx1, cid in enumerate(data["course_code"]):
        #index = data[data['userID'] == user].index.values[0]
        idx2 = reversed_item_dict_one_hot[cid] # cid to idx
        cid2 = data["course_code"][idx2]
        cat_f = data["category"][idx2]
        level_f = data["level"][idx2]
        row1 = [cid2, cat_f]
        row2 = [cid2, str(level_f)]
        cat_f_all[idx2] = row1
        level_f_all[idx2] = row2

    cat_f_all_sorted = dict(sorted(cat_f_all.items(), key=lambda item: item[0], reverse=False))
    level_f_all_sorted = dict(sorted(level_f_all.items(), key=lambda item: item[0], reverse=False))
    cat_f_all = list(cat_f_all_sorted.values())
    level_f_all = list(level_f_all_sorted.values())
    
    cat_f_all_df = pd.DataFrame(cat_f_all, columns=['item', 'cat_f'])
    level_f_all_df = pd.DataFrame(level_f_all, columns=['item', 'level_f'])
    one_hot_encoded_cat, one_hot_df_cat, item_dict_one_hot_cat, cat_dict_one_hot = convert_to_one_hot_encoding_cat(cat_f_all_df)
    one_hot_encoded_level, one_hot_df_level, item_dict_one_hot_level, level_dict_one_hot = convert_to_one_hot_encoding_level(level_f_all_df)

    reversed_dict_cat_to_idx = dict(zip(cat_dict_one_hot.values(), cat_dict_one_hot.keys()))  # cat, idx
    reversed_dict_level_to_idx = dict(zip(level_dict_one_hot.values(), level_dict_one_hot.keys()))  # level, idx

    return one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx,  reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level


class GAT(torch.nn.Module):
    def __init__(self, num_users, num_items, num_fet, embedding_dim, num_layers, edge_dropout, node_dropout, num_heads=1, seed_value = 42):
        super(GAT, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fet = num_fet
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.edge_dropout = edge_dropout
        self.node_dropout = node_dropout
        self.num_heads = num_heads

        torch.manual_seed(seed_value)  # Ensure same random initialization
        # Initialize user and item embeddings
        self.item_embeddings = torch.nn.Embedding(self.num_items, embedding_dim)
        self.user_embeddings = torch.nn.Embedding(self.num_users, embedding_dim)
        self.fet_embeddings = torch.nn.Embedding(self.num_fet, embedding_dim)
        torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.fet_embeddings.weight)
        print("init emb: ", self.item_embeddings.weight[:2])  # Print first 5 embeddings

        # GAT layers
        in_channels = out_channels = hidden_channels = embedding_dim
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=edge_dropout)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=edge_dropout)
        self.gat3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=edge_dropout)
        #self.linear_transform = torch.nn.Linear(hidden_channels * num_heads, embedding_dim)

    def forward(self, edge_index, edge_weight=None):
        x = torch.cat([self.item_embeddings.weight, self.user_embeddings.weight, self.fet_embeddings.weight], dim=0)
        all_embeddings = [x]
        attention_weights = []  # List to store attention weights
        edge_index_out_all = []

        if self.num_layers > 1:
            # First GAT layer
            x, (edge_index_out, alpha) = self.gat1(x, edge_index, edge_weight, return_attention_weights=True)
            attention_weights.append(alpha)
            edge_index_out_all.append(edge_index_out)
            all_embeddings.append(x)

            for _ in range(self.num_layers - 2):
                x = F.relu(x)
                x = F.dropout(x, p=self.node_dropout, training=self.training)
                x, (edge_index_out, alpha) = self.gat2(x, edge_index, return_attention_weights=True)
                attention_weights.append(alpha)
                edge_index_out_all.append(edge_index_out)
                all_embeddings.append(x)

            # Final GAT layer
            x = F.relu(x)
            x = F.dropout(x, p=self.node_dropout, training=self.training)
            x, (edge_index_out, alpha) = self.gat3(x, edge_index, return_attention_weights=True)
            attention_weights.append(alpha)
            edge_index_out_all.append(edge_index_out)
            all_embeddings.append(x)

        else:
            # Single-layer GAT
            x, (edge_index_out, alpha) = self.gat1(x, edge_index, edge_weight, return_attention_weights=True)
            attention_weights.append(alpha)
            all_embeddings.append(x)
            edge_index_out_all.append(edge_index_out)

        # Aggregate all embeddings
        if self.num_heads == 1:
            final_embedding = torch.mean(torch.stack(all_embeddings, dim=0), dim=0) # avg over emb of all attention layers
            #final_embedding = x
        else:
            all_emb = [all_embeddings[0]]
            for emb in all_embeddings[1:]:
                if emb.size(-1) != self.embedding_dim:
                    emb = emb.view(emb.size(0), self.num_heads, -1)
                    emb = emb.mean(dim=1)
                all_emb.append(emb)
            final_embedding = torch.mean(torch.stack(all_emb, dim=0), dim=0)  # avg over emb of all attention layers
            #final_embedding = all_emb[-1]
        #final_embedding = x
        # Split final embedding
        item_embedding = final_embedding[:self.num_items]
        user_embedding = final_embedding[self.num_items:self.num_items + self.num_users]
        fet_embedding = final_embedding[self.num_items + self.num_users:]

        return item_embedding, user_embedding, fet_embedding, final_embedding, attention_weights, edge_index_out_all

    
    def bpr_loss(self, user_embeddings, pos_item_embeddings, neg_item_embeddings):
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
        # pos_f_scores = torch.sum(user_embeddings * pos_f_embeddings, dim=1)
        # neg_f_scores = torch.sum(user_embeddings * neg_f_embeddings, dim=1)
        # return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, one_hot_encoded_text, n_layers, embedding_dim, n_epochs, l_rate, edge_dropout, node_dropout, heads, threshold_weight_edges_iw,  threshold_weight_edges_ww, seed_value, version):
    # Step 1: Data Preparation (Assume a small dataset)
    num_users = one_hot_encoded_train.shape[0]
    num_items = one_hot_encoded_train.shape[1]
    num_cat_f = one_hot_encoded_cat.shape[1] 
    num_level_f = one_hot_encoded_level.shape[1]
    num_text_f = one_hot_encoded_text.shape[1]
    # num_fet = one_hot_encoded_cat.shape[1] + one_hot_encoded_level.shape[1]
    num_fet = num_cat_f + num_level_f + num_text_f

    interaction_matrix = np.array(one_hot_encoded_train)
    adj_matrix_all = np.zeros((num_items+num_users+num_fet, num_items+num_users+num_fet))
    edge_index = []
    # user item edges
    cnt4 = 0
    for user in range(num_users):
        for item in range(num_items):
            if interaction_matrix[user, item] == 1:
                 cnt4+= 1
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, num_items + user]) 
                 adj_matrix_all[item, num_items + user] = 1
                 #adj_matrix_all[num_items + user, item] = 1
    cnt1= 0
   
    interaction_matrix_cat = np.array(one_hot_encoded_cat)
    for item in range(num_items):
        for cat in range(num_cat_f):
            if interaction_matrix_cat[item, cat] == 1:
                 cnt4+= 1
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, (num_items + num_users+ cat)]) 
                 adj_matrix_all[item, num_items + num_users+ cat] = 1
    
    interaction_matrix_level = np.array(one_hot_encoded_level)
    for item in range(num_items):
        for level in range(num_level_f):
            if interaction_matrix_level[item, level] == 1:
                 cnt4 += 1
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, (num_items + num_users+ num_cat_f+level)]) 
                 adj_matrix_all[item, num_items + num_users+ num_cat_f+level] = 1
    cnt2= 0
    interaction_matrix_text = np.array(one_hot_encoded_text)
    kept_word_node_to_idx = {}
    for item in range(num_items):
        for word_idx in range(num_text_f):
            if interaction_matrix_text[item, word_idx] >= threshold_weight_edges_iw: # threshold, th = 0.1
            #if interaction_matrix_text[item, word_idx] == 1: # threshold, th = 0.2
                 cnt2+= 1
                 kept_word_node_to_idx[num_items + num_users+ num_cat_f + num_level_f+ word_idx] = word_idx  # word node to word idx
                # edge_index.append([user, num_users + item])  # User connected to Item
                 edge_index.append([item, (num_items + num_users+ num_cat_f+num_level_f+ word_idx)]) 
                #  adj_matrix_all[item, num_items + num_users+ num_cat_f+num_level_f+ word_idx] = 1
                 adj_matrix_all[item, num_items + num_users+ num_cat_f+num_level_f+ word_idx] = interaction_matrix_text[item, word_idx]
    # # word word edges
    # cnt3 = 0
    # for word1 in range(num_items, num_items+ num_text_f):
    #     for word2 in range(num_text_f):
    #         if interaction_matrix_text[word1, word2] > threshold_weight_edges_ww and (word1-num_items) != word2: # normalize these values
    #              cnt3 += 1
    #             # edge_index.append([user, num_users + item])  # User connected to Item
    #              edge_index.append([(num_items + num_users+ num_cat_f + num_level_f+ word1-num_items), (num_items + num_users+ num_cat_f + num_level_f+ word2)]) 
    #              adj_matrix_all[num_items + num_users+ num_cat_f + num_level_f+ word1-num_items, num_items + num_users+ num_cat_f + num_level_f+ word2] = threshold_weight_edges_ww
    #             # adj_matrix_all[num_items + num_users+ word2, num_items + num_users+ word1-num_items] = interaction_matrix_text[word1, word2] # comment these connections if necessary
    # print("num of item-item edges: ", cnt1) # num of item-item sequential edges 
    print("num of item-word edges: ", cnt2) # num of item-word edges
    # print("num of item-word edges: ", cnt3) # num of word-word edges
    #print("num of edges: ", cnt4)

    #edge_index = torch.tensor(edge_index).t().contiguous()
    # word_cooccurrence = np.zeros((vocab_size, vocab_size))

    adj_sparse = sp.coo_matrix(adj_matrix_all)

    # Step 3: Convert the sparse matrix to a PyTorch Geometric Data object
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_sparse)
    # #Embeddings for users and items
    torch.manual_seed(seed_value)  # Ensure same random initialization
    user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
    item_embeddings = torch.nn.Embedding(num_items, embedding_dim)
    fet_embeddings = torch.nn.Embedding(num_fet, embedding_dim)

    torch.nn.init.xavier_uniform_(user_embeddings.weight)
    torch.nn.init.xavier_uniform_(item_embeddings.weight)
    torch.nn.init.xavier_uniform_(fet_embeddings.weight)
    
    x = torch.cat([item_embeddings.weight, user_embeddings.weight, fet_embeddings.weight], dim=0)
    # data = Data(x=x, edge_index=edge_index)
    data = Data(x=x, edge_index=edge_index, edge_attr= edge_weight)
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
     # Convert the matrix to a dictionary
    user_interactions = {
        user: {item for item, interacted in enumerate(row) if interacted}
        for user, row in enumerate(interaction_matrix)
    }

    # Step 3: Instantiate the model
    # model = LightGCN(num_users, num_items, num_fet, embedding_dim, n_layers, dropout).to(device)
    
    model = GAT(num_users, num_items, num_fet, embedding_dim, n_layers, edge_dropout, node_dropout, heads, seed_value).to(device)
   

    optimizer = optim.Adam(model.parameters(), lr=l_rate)

    # Step 4: Training Loop
    #final_x = data.x
    epochs = n_epochs
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        item_embeddings, user_embeddings, fet_embeddings, final_x, final_attention_weights_all_layers, final_edge_index_all_layers = model(edge_index.to(device), edge_weight.to(device)) 
        
        user_ids = torch.randint(0, num_users, (num_users,), device=device)
        pos_item_ids = []
        neg_item_ids = []
        # pos_fet_ids = []
        # neg_fet_ids = []

        for user_id in user_ids:
            # Sample a positive item from the user's actual interactions
            pos_item = random.choice(list(user_interactions[user_id.item()]))  
            pos_item_ids.append(pos_item)
            
            # Sample a negative item not in the user's interactions
            neg_item = random.choice([item for item in range(num_items) if item not in list(user_interactions[user_id.item()])])
            neg_item_ids.append(neg_item)

        # Convert lists to tensors
        pos_item_ids = torch.tensor(pos_item_ids, device=device)
        neg_item_ids = torch.tensor(neg_item_ids, device=device)
        # pos_fet_ids = torch.tensor(pos_fet_ids, device=device)
        # neg_fet_ids = torch.tensor(neg_fet_ids, device=device)

        # Gather embeddings for BPR Loss
        u_emb = user_embeddings[user_ids]
        pos_i_emb = item_embeddings[pos_item_ids]
        neg_i_emb = item_embeddings[neg_item_ids]
        # pos_f_emb = fet_embeddings[pos_fet_ids]
        # neg_f_emb = fet_embeddings[neg_fet_ids]

        # Compute BPR Loss
        loss = model.bpr_loss(u_emb, pos_i_emb, neg_i_emb)
        
        loss.backward()
        optimizer.step()
        
        # if epoch % 20 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        #data.x = torch.cat([user_embeddings.weight.detach(), item_embeddings.weight.detach()], dim=0)

    # print("from trained model, Item embeddings of 0: ", item_embeddings[0])
    # print("from trained model, user_embeddings of last user: ", user_embeddings[num_users-1])
    base_path = './saved_model_GAT_CDREAM_uis_text_GRU_avg_attn_layers_v3_keywords_tfidf/'
    os.makedirs(base_path, exist_ok=True)
    model_filename = f"model_v{version}.pth"
    full_path = os.path.join(base_path, model_filename)
    torch.save(model.state_dict(), full_path)

    print(final_x.shape)  
    print("num of nodes: ", final_x.shape[0])
    print("num of edges: ", cnt4)
    print("len of attention weights: ", len(final_attention_weights_all_layers))
    print("num of edges in attention weights: ", len(final_attention_weights_all_layers[0]))
    print("shape of edge index out: ", final_edge_index_all_layers[0].shape)
    return model, data, final_x, final_attention_weights_all_layers, final_edge_index_all_layers, kept_word_node_to_idx

if __name__ == '__main__':
    train_data = pd.read_json(./train_data_all.json', orient='records', lines= True)
    start = time.time()
    print("Start")
    dataTrain, dataTest, dataTotal, item_list, item_dict, reversed_item_dict, one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, reversed_item_dict_one_hot, user_dict_one_hot, one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train = preprocess_data(train_data)
    
    num_items = one_hot_encoded_train.shape[1]
    df = pd.read_json('./course_info_desc_keywords.json', orient='records', lines=True)
    #course_concept_path = "./course_concepts_v2.csv"
    one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx, reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level = convert_side_info_to_one_hot_encoding(df, reversed_item_dict_one_hot, num_items)
    window_s = 10
    adj_mat, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx = get_adj_matrix_text_v3(df, reversed_item_dict_one_hot, window_s)
    #print(adj_mat)
    #print(adj_mat.shape)
    print(vocab_size)
    dense_adj_matrix_text = adj_mat.toarray()
    adj_matrix_text = normalize(dense_adj_matrix_text, norm='l2')
    #print(adj_matrix_text)
    threshold_weight_edges_iw = 0.1
    threshold_weight_edges_ww = 0.1

    n_layers = 3
    embedding_dim = 128
    epochs = 200
    lr = 0.01
    edge_dropout = 0
    node_dropout = 0
    n_heads = 2
    version = 1
    seed_value = 42
    
    model, data, final_x, final_attention_weights_all_layers, final_edge_index_all_layers, kept_word_node_to_idx = train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, adj_matrix_text, n_layers, embedding_dim, epochs, lr, edge_dropout, node_dropout, n_heads, threshold_weight_edges_iw,  threshold_weight_edges_ww, seed_value, version)
    end = time.time()
    print("time: ", end-start)
