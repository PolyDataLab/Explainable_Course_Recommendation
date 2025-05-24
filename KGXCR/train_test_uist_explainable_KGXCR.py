import os
import math
import random
import time
import logging
import pickle
#from turtle import shapesize
import torch
import numpy as np
from math import ceil
import data_helpers as dh
from config import Config  # scp
from rnn_model_GPU import DRModel
import tensorflow as tf
from dataprocess_v1 import *
#from utils import *
import utils
from utils import *
#from offered_courses import *
import pandas as pd
import preprocess_for_GAT
from offered_courses_v2 import *
#from topic_model_v2 import *
import math
from training_GAT_uist import * # type: ignore  # scp
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import networkx as nx
from itertools import islice
import random
from sklearn.metrics import recall_score
import torch.nn.functional as F
import copy

# Set seed value
seed_value = 42 

# Set seed for Python's random module
random.seed(seed_value)

# Set seed for NumPy
np.random.seed(seed_value)

# Set seed for PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # If using multi-GPU
torch.use_deterministic_algorithms(True)  # Ensures deterministic behavior in PyTorch >=1.8

# Ensure deterministic behavior in cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Set environment variable for more deterministic behavior
os.environ["PYTHONHASHSEED"] = str(seed_value)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


logging.info("✔︎ CDREAM_LGCN Model Training...")
logger = dh.logger_fn("torch-log", "logs/training-{0}.log".format(time.asctime()))

dilim = '-' * 120
logger.info(dilim)
for attr in sorted(Config().__dict__):
    logger.info('{:>50}|{:<50}'.format(attr.upper(), Config().__dict__[attr]))
logger.info(dilim)

def path_generation(attention_weights_tt, edge_index_tt, item_dict_one_hot, user_dict_one_hot, cat_dict_one_hot, level_dict_one_hot, vocab_dict_idx_to_wrd, kept_word_node_to_idx, num_items, num_users, num_cat_f, num_level_f, num_text_f, df, reversed_item_dict_one_hot, s1, d1):
    edge_index = edge_index_tt.detach().numpy()
    attention_weights_np = attention_weights_tt.detach().numpy()

    offset = abs(np.min(attention_weights_np)) + 0.001  # Add small epsilon for stability
    attention_weights_np1 = attention_weights_np + offset

    # Reconstruct graph
    #G = nx.DiGraph()  # Directed graph; use nx.Graph() for undirected
    G = nx.Graph() 

    # Invert weights to represent "distance"
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0, i], edge_index[1, i]
        # weight = attention_weights[i]

        #avg_attention_weights_np = attention_weights_np.mean(axis=1)
        if len(attention_weights_np1[i]) >= 2:
            #print(f"Error: attention_weights_np1[{i}] has fewer than 2 elements.")
            weight1 = attention_weights_np1[i][0] # weight for src node to destination node
            weight2 = attention_weights_np1[i][1] # weight for target node to source node
        else:
            weight1 = attention_weights_np1[i][0]
            weight2 = weight1
        if src<=617 and tgt<=617: # item-item edge
            avg_weight = weight1 # item1--> item2 directed edge
        else:
            avg_weight = (weight1+weight2)/2.0 # undirected edge (avg attention)
        # Convert to distance (higher attention -> lower distance)
        distance = 1 / avg_weight  # Assuming weights > 0
        #distance = 1 - avg_weight  
        if int(src)==s1 and int(tgt)==d1: 
            G.add_edge(src, tgt, weight=float(10000000000000))
        elif int(src)==d1 and int(tgt)==s1:
            G.add_edge(src, tgt, weight=float(10000000000000))
        elif int(src)< num_items and (int(tgt)>= num_items and int(tgt)< num_items+num_users):
            G.add_edge(src, tgt, weight=float(10000000000000))  # keeping highest distance between item-user connections to avoid collaborative explanations
        elif int(tgt)< num_items and (int(src)>= num_items and int(src)< num_items+num_users):
            G.add_edge(src, tgt, weight=float(10000000000000))  # keeping highest distance between user-item connections to avoid collaborative explanations
        else:
            G.add_edge(src, tgt, weight=float(distance))

    # Compute the shortest path between node 0 and node 618
    try:
        # Compute all shortest paths
        all_paths = nx.all_shortest_paths(G, source=s1, target=d1, weight='weight')
        # Calculate the total weight for each path and sort
        sorted_paths_by_weights = sorted(
            all_paths,
            key=lambda path: sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
        )
        # Filter paths with at least 2 hops
        valid_paths = [path for path in sorted_paths_by_weights if len(path) > 2]

        if valid_paths:
            # Select one of the valid paths (e.g., the first one)
            optimal_path = valid_paths[0]
            # Compute the path length
            path_length = sum(G[optimal_path[i]][optimal_path[i + 1]]['weight'] for i in range(len(optimal_path) - 1))
            # print(f"Shortest Path with at least 2 hops: {optimal_path}")
            # print(f"Path Length: {path_length}")
        else:
            optimal_path = sorted_paths_by_weights[0]
            #optimal_path = top_paths[0]
            path_length = sum(G[optimal_path[i]][optimal_path[i + 1]]['weight'] for i in range(len(optimal_path)-1))
        
        #print(f"Shortest Path: {path}")  # use the actual userIDs, itemIDs, and item features to define the path
        path_main = []
        for idx1 in optimal_path:
            if idx1<num_items:
                #path_main.append(item_dict_one_hot[idx1])
                cid = item_dict_one_hot[idx1]
                idx2 = df[df['course_code']==cid].index.values[0]
                # index1 = dataTest_prev[dataTest_prev['userID'] == uid_sample].index.values[0]
                cname = df["course_name"][idx2]
                #text_f = data["course_description"][idx2]
                path_main.append(cname)
        
            elif idx1<num_items+num_users:
                path_main.append(user_dict_one_hot[idx1-num_items])
            elif idx1<num_items+num_users+num_cat_f:
                path_main.append(cat_dict_one_hot[idx1-num_items-num_users])
            elif idx1<num_items+num_users+num_cat_f+num_level_f:
                path_main.append(level_dict_one_hot[idx1-num_items-num_users-num_cat_f])
            elif idx1<num_items+num_users+num_cat_f+num_level_f+num_text_f:
                path_main.append(vocab_dict_idx_to_wrd[kept_word_node_to_idx[idx1]])
        # print(f"Shortest Path main: {path_main}")
        # print(f"Path Length: {path_length}")   
    except nx.NetworkXNoPath:
        print("No path exists between the specified nodes.")
    return optimal_path, path_main, path_length


def path_generation_collab(attention_weights_tt, edge_index_tt, item_dict_one_hot, user_dict_one_hot, cat_dict_one_hot, level_dict_one_hot, vocab_dict_idx_to_wrd, kept_word_node_to_idx, num_items, num_users, num_cat_f, num_level_f, num_text_f, df, reversed_item_dict_one_hot, s1, d1):
    edge_index = edge_index_tt.detach().numpy()
    attention_weights_np = attention_weights_tt.detach().numpy()

    # Step 1: Add positive offset to make weights strictly positive
    offset = abs(np.min(attention_weights_np)) + 0.001  # Add small epsilon for stability
    attention_weights_np1 = attention_weights_np + offset

    # Reconstruct graph
    #G = nx.DiGraph()  # Directed graph; use nx.Graph() for undirected
    G = nx.Graph() 

    # Invert weights to represent "distance"
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0, i], edge_index[1, i]
        if len(attention_weights_np1[i]) >= 2:
            #print(f"Error: attention_weights_np1[{i}] has fewer than 2 elements.")
            weight1 = attention_weights_np1[i][0] # weight for src node to destination node
            weight2 = attention_weights_np1[i][1] # weight for target node to source node
        else:
            weight1 = attention_weights_np1[i][0]
            weight2 = weight1
        if src<=617 and tgt<=617: # item-item edge
            avg_weight = weight1 # item1--> item2 directed edge
        else:
            avg_weight = (weight1+weight2)/2.0 # undirected edge (avg attention)
        # Convert to distance (higher attention -> lower distance)
        distance = 1 / avg_weight  # Assuming weights > 0
        #distance = 1 - avg_weight  
        if int(src)==s1 and int(tgt)==d1: 
            G.add_edge(src, tgt, weight=float(10000000000000))
        elif int(src)==d1 and int(tgt)==s1:
            G.add_edge(src, tgt, weight=float(10000000000000))
        # elif int(src)< num_items and (int(tgt)>= num_items and int(tgt)< num_items+num_users):
        #     G.add_edge(src, tgt, weight=float(10000000000000))  # keeping highest distance between item-user connections to avoid collaborative explanations
        # elif int(tgt)< num_items and (int(src)>= num_items and int(src)< num_items+num_users):
        #     G.add_edge(src, tgt, weight=float(10000000000000))  # keeping highest distance between user-item connections to avoid collaborative explanations
        else:
            G.add_edge(src, tgt, weight=float(distance))

    # Compute the shortest path between node 0 and node 618
    try:
        # Compute all shortest paths
        flag = 0
        while flag==0:
            all_paths = nx.all_shortest_paths(G, source=s1, target=d1, weight='weight')
            # Calculate the total weight for each path and sort
            sorted_paths_by_weights = sorted(
                all_paths,
                key=lambda path: sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
            )
            # Filter paths with at least 2 hops
            valid_paths = [path for path in sorted_paths_by_weights if len(path) > 2]

            if valid_paths:
                # Select one of the valid paths (e.g., the first one)
                optimal_path = valid_paths[0]
                # Compute the path length
                path_length = sum(G[optimal_path[i]][optimal_path[i + 1]]['weight'] for i in range(len(optimal_path) - 1))
                # print(f"Shortest Path with at least 2 hops: {optimal_path}")
                # print(f"Path Length: {path_length}")
            else:
                optimal_path = sorted_paths_by_weights[0]
                #optimal_path = top_paths[0]
                path_length = sum(G[optimal_path[i]][optimal_path[i + 1]]['weight'] for i in range(len(optimal_path)-1))
                # print(f"Shortest Path with 1 hop: {optimal_path}")
                # print(f"Path Length: {path_length}")
            
            #print(f"Shortest Path: {path}")  # use the actual userIDs, itemIDs, and item features to define the path
            path_main = []
            #flag = 0
            for idx1 in optimal_path:
                if idx1<num_items:
                    #path_main.append(item_dict_one_hot[idx1])
                    cid = item_dict_one_hot[idx1]
                    idx2 = df[df['course_code']==cid].index.values[0]
                    # index1 = dataTest_prev[dataTest_prev['userID'] == uid_sample].index.values[0]
                    cname = df["course_name"][idx2]
                    #text_f = data["course_description"][idx2]
                    path_main.append(cname)
            
                elif idx1<num_items+num_users:
                    path_main.append(user_dict_one_hot[idx1-num_items])
                    flag = 1
                elif idx1<num_items+num_users+num_cat_f:
                    path_main.append(cat_dict_one_hot[idx1-num_items-num_users])
                elif idx1<num_items+num_users+num_cat_f+num_level_f:
                    path_main.append(level_dict_one_hot[idx1-num_items-num_users-num_cat_f])
                elif idx1<num_items+num_users+num_cat_f+num_level_f+num_text_f:
                    path_main.append(vocab_dict_idx_to_wrd[kept_word_node_to_idx[idx1]])
            if flag==0: 
                for idx3 in range(len(optimal_path)-1):
                    G.add_edge(optimal_path[idx3], optimal_path[idx3+1], weight=float(10000000000000))

        # print(f"Shortest Path main: {path_main}")
        # print(f"Path Length: {path_length}")   
    except nx.NetworkXNoPath:
        print("No path exists between the specified nodes.")
    # if flag==1:
    return optimal_path, path_main, path_length

def recall_cal_2(positives, pred_items):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        #if(correct_preds>=1): count_at_least_one_cor_pred += 1
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))
        return float(correct_preds/actual_bsize)

def dcg_at_k_v2(predicted, ground_truth, k): # for only one explained item
    """
    Compute DCG@K.
    Parameters:
        predicted (list): List of predicted items.
        ground_truth (set): Set of ground-truth relevant items.
        k (int): Rank position to calculate DCG.
    Returns:
        float: DCG value.
    """
    dcg = 0.0
    #for i in range(k):
        #if i < len(predicted):
            #p_k = 1 if predicted[i] in ground_truth else 0 # if i-th item we recommend is relevant or not
    p_k = 1 # if predicted[i] in ground_truth else 0 # if i-th item we recommend is relevant or not
    dcg += p_k / np.log2(predicted + 1)  # log2(predicted+1) because index starts at 1 (i+1 when indexing from 0)
    return dcg

def idcg_at_k_v2(ground_truth, k):
    """
    Compute IDCG@K (ideal DCG).
    Parameters:
        ground_truth (set): Set of ground-truth relevant items.
        k (int): Rank position to calculate IDCG.
    Returns:
        float: IDCG value.
    """
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return ideal_dcg

def ndcg_at_k_v2(predicted, ground_truth, k):
    """
    Compute NDCG@K.
    Parameters:
        predicted (list): List of predicted items.
        ground_truth (set): Set of ground-truth relevant items.
        k (int): Rank position to calculate NDCG.
    Returns:
        float: NDCG value.
    """
    ndcg = 0
    for pr in range(len(predicted)):
        pr_rank = predicted[pr]
        k= len(ground_truth[pr])
        dcg = dcg_at_k_v2(pr_rank, ground_truth, k) # not considering if other items are relevant or not. actually setting them 0, setting 1 for explained item
        idcg = idcg_at_k_v2(ground_truth, k)
        ndcg+= dcg / idcg if idcg > 0 else 0.0
    return ndcg/len(predicted)
def dcg_at_k(predicted, ground_truth, k):
    """
    Compute DCG@K.
    Parameters:
        predicted (list): List of predicted items.
        ground_truth (set): Set of ground-truth relevant items.
        k (int): Rank position to calculate DCG.
    Returns:
        float: DCG value.
    """
    dcg = 0.0
    for i in range(k):
        if i < len(predicted):
            p_k = 1 if predicted[i] in ground_truth else 0 # if i-th item we recommend is relevant or not
            dcg += p_k / np.log2(i + 2)  # log2(i+1+1) because index starts at 0
    return dcg

def idcg_at_k(ground_truth, k):
    """
    Compute IDCG@K (ideal DCG).
    Parameters:
        ground_truth (set): Set of ground-truth relevant items.
        k (int): Rank position to calculate IDCG.
    Returns:
        float: IDCG value.
    """
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return ideal_dcg

def ndcg_at_k(predicted, ground_truth, k):
    """
    Compute NDCG@K.
    Parameters:
        predicted (list): List of predicted items.
        ground_truth (set): Set of ground-truth relevant items.
        k (int): Rank position to calculate NDCG.
    Returns:
        float: NDCG value.
    """
    k= len(ground_truth)
    dcg = dcg_at_k(predicted, ground_truth, k)
    idcg = idcg_at_k(ground_truth, k)
    return dcg / idcg if idcg > 0 else 0.0

def cal_similarity_score(p_item, item1, item_embedding_all):
    """
    Compute cosine similarity between two items using their embeddings.

    Args:
        p_item (int): Index of the first item.
        item1 (int): Index of the second item.
        item_embedding_all (torch.Tensor): A tensor containing embeddings for all items.

    Returns:
        float: Cosine similarity score between the two items.
    """
    # Get the embeddings of the two items
    emb1 = item_embedding_all[p_item]  # Embedding for p_item
    emb2 = item_embedding_all[item1]   # Embedding for item1

    # Compute cosine similarity
    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))

    return similarity.item()  # Convert tensor to Python float

def add_source_to_dict(lst, source_name):
    return [{**d, "source": source_name} for d in lst]

def fidelity_plus_measure_main_v2(ground_truth_items_all, original_recommended_items_all, perturbed_recommended_items_all, rank_explained_rec_item_original, rank_explained_rec_item_perturbed):
    # Sample 100 users
    #sampled_users = random.sample(test_data.keys(), num_samples)
    original_recalls = []
    perturbed_recalls = []

    for idx1 in range(len(ground_truth_items_all)):
        # if idx1 ==100: break
        original_recall = recall_cal_2(ground_truth_items_all[idx1], original_recommended_items_all[idx1])
        perturbed_recall = recall_cal_2(ground_truth_items_all[idx1], perturbed_recommended_items_all[idx1])

        original_recalls.append(original_recall)
        perturbed_recalls.append(perturbed_recall)
    
    # Compute fidelity as the average difference in recall per instance
    sum1 =0
    for rec in original_recalls:
        sum1 += rec
    avg_original_recalls= sum1/len(original_recalls)

    sum2 =0
    for rec in perturbed_recalls:
        sum2 += rec
    avg_perturbed_recalls= sum2/len(perturbed_recalls)

    items_ranks_original = list(rank_explained_rec_item_original.values())
    items_ranks_perturbed = list(rank_explained_rec_item_perturbed.values())
    dif1 = 0
    sum_rank_org = 0
    sum_rank_per = 0
    for idx1 in range(len(items_ranks_original)):
        if idx1 ==100: break
        item1, rank_org = items_ranks_original[idx1] # a list of item, rank
        item2, rank_per = items_ranks_perturbed[idx1] # a list of item, rank
        dif1_rank = rank_per- rank_org # positive value = better, low rank = lower index = better recommendation ranking
        dif1 += dif1_rank
        sum_rank_org += rank_org
        sum_rank_per += rank_per
    
    avg_rank_org = sum_rank_org/ 100
    avg_rank_per = sum_rank_per/ 100 # expectation = it should be higher 
    fidelity_rank = dif1/ 100
    ranks_original = [rank for item, rank in items_ranks_original]
    ranks_perturbed = [rank for item, rank in items_ranks_perturbed]
    ranks_original = ranks_original[:100]
    ranks_perturbed = ranks_perturbed[:100]

    # Compute NDCG and Fidelity
    k = 5  # Cut-off rank for NDCG
    ground_truth_items_all = ground_truth_items_all[:100]
    original_recommended_items_all= original_recommended_items_all[:100]
    perturbed_recommended_items_all= perturbed_recommended_items_all[:100]
    ndcg_original_explained_item = ndcg_at_k_v2(ranks_original, ground_truth_items_all, k)
    ndcg_perturbed_explained_item = ndcg_at_k_v2(ranks_perturbed, ground_truth_items_all, k)

    ndcg1 = 0
    ndcg2= 0
    for idx2 in range(len(ground_truth_items_all)):
        if idx2==100: break
        ndcg1 += ndcg_at_k(original_recommended_items_all[idx2], ground_truth_items_all[idx2], k)
        ndcg2 += ndcg_at_k(perturbed_recommended_items_all[idx2], ground_truth_items_all[idx2], k)

    ndcg_original_all_rec_items = ndcg1/ len(original_recommended_items_all)
    ndcg_perturbed_all_rec_items = ndcg2/ len(original_recommended_items_all)
    #ndcg_original_all_rec_items, ndcg_perturbed_all_rec_items = 0, 0
    fidelity_plus = sum(
        original - perturbed
        for original, perturbed in zip(original_recalls, perturbed_recalls)
    ) / len(original_recalls)

    return fidelity_plus, avg_original_recalls, avg_perturbed_recalls, fidelity_rank, avg_rank_org, avg_rank_per, ndcg_original_explained_item, ndcg_perturbed_explained_item, ndcg_original_all_rec_items, ndcg_perturbed_all_rec_items


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
    
    # # Step 3: Combine TF-IDF and PMI into adjacency matrix
    tfidf_csr = csr_matrix(tfidf_matrix)

    # Create an adjacency matrix of the required shape
    adjacency_matrix = csr_matrix((tfidf_matrix.shape[0]+vocab_size, vocab_size))

    # Fill in document-word (TF-IDF)
    adjacency_matrix[:tfidf_matrix.shape[0], :] = tfidf_csr  # Documents to Words (TF-IDF)

    # Fill in word-word (PMI)
    # adjacency_matrix[:vocab_size, :] = csr_matrix(pmi_matrix)  # Words to Words (PMI)
    adjacency_matrix[tfidf_matrix.shape[0]:, :] = csr_matrix(pmi_matrix)  # Words to Words (PMI)

    return adjacency_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx

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
        #row1 = [cname2, text_f]
        # row1 = [cname2, concept_f]
        row1 = [cid2, concept_f]
        concept_f_all[idx2] = row1
       
    concept_f_all_sorted = dict(sorted(concept_f_all.items(), key=lambda item: item[0], reverse=False))
    concept_f_all_new = list(concept_f_all_sorted.values())
    
    item_descriptions = []
    #item_names = []
    for list1 in concept_f_all_new:
        cid3, cconcept = list1
        cdesc = ' '.join(cconcept)
        item_descriptions.append(cdesc)
        #item_names.append(cname3)

    adj_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx = compute_tfidf_pmi(item_descriptions, window_size=window_s)
    #adj_mat, vocab_size = compute_tfidf_pmi(item_concepts, window_size=window_s)
    return adj_matrix, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx


def update_idx_of_item_embedding(one_hot_encoded_data, num_users_train, item_embeddings, item_dict_idx_to_cid, item_dict_cid_to_idx, item_dict_one_hot):
    # num_nodes = data.x.size(0)
    
    #interaction_matrix = np.array(one_hot_encoded_data)
    item_embeddings2 = torch.zeros_like(item_embeddings, dtype= torch.float32)
    #item_embeddings2 = torch.zeros_like(item_embeddings)
    for idx1, it_embed in enumerate(item_embeddings):
        cid = item_dict_one_hot[idx1] # idx to cid in LightGCN
        idx2 = item_dict_cid_to_idx[cid] # cid to idx in training data
        item_embeddings2[idx2] = it_embed

    return item_embeddings2

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

def train(offered_courses, train_set_without_target, target_set, item_dict, item_embeddings, train_data_all, valid_set_without_target, n_rnn_lays, rnn_drops, rnn_lr, device):
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    #train_data = dh.load_data(Config().TRAININGSET_DIR)
    train_data = train_data_all

    logger.info("✔︎ Validation data processing...")
    #validation_data = dh.load_data(Config().VALIDATIONSET_DIR)
    validation_data = valid_set_without_target

    logger.info("✔︎ Test data processing...")
    #target_data = dh.load_data(Config().TESTSET_DIR)
    target_data = target_set

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Model config
    #model = DRModel(Config(), item_embeddings, n_rnn_lays, rnn_drops, rnn_lr).to(device)
    model = DRModel(Config(), item_embeddings, n_rnn_lays, rnn_drops, rnn_lr, device = device)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=rnn_lr)

    def bpr_loss(uids, baskets, dynamic_user, item_embedding, neg_samples, device):
        # """
        # Bayesian personalized ranking loss for implicit feedback.
        # Args:
        #     uids: batch of users' IDs (tensor)
        #     baskets: batch of users' baskets (list of lists of item indices)
        #     dynamic_user: batch of users' dynamic representations (tensor on device)
        #     item_embedding: item_embedding matrix (tensor on device)
        #     neg_samples: dictionary of negative samples for each user
        #     device: torch device (e.g., 'cuda' or 'cpu')
        # """
        loss = torch.tensor(0.0, device=device)  # Initialize loss on the correct device
        for uid, bks, du in zip(uids, baskets, dynamic_user):
            du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item], on device
            loss_u = []  # Loss for each user
            
            for t, basket_t in enumerate(bks):
                if basket_t[0] != 0 and t != 0:
                    # Positive indices (basket items)
                    pos_idx = torch.tensor(basket_t).to(device)  # Move to device

                    # Sample negative products
                    neg = random.sample(list(neg_samples[uid.item()]), len(basket_t))
                    # neg_idx = torch.LongTensor(neg).to(device)  # Move to device
                    neg_idx = torch.tensor(neg).to(device)  # Move to device

                    # Score p(u, t, v > v')
                    score = du_p_product[t - 1][pos_idx] - du_p_product[t - 1][neg_idx]

                    # Average Negative log likelihood for basket_t
                    loss_u.append(torch.mean(-torch.nn.functional.logsigmoid(score)))

            if loss_u:  # Avoid division by zero if no loss components
                loss += sum(loss_u) / len(loss_u)

        avg_loss = loss / len(baskets)  # Average loss over all users in the batch
        return avg_loss

    def train_model():
        model.train()  # turn on training mode for dropout
        dr_hidden = model.init_hidden(Config().batch_size).to(device)
        train_loss = 0
        #train_recall = 0.0
        start_time= time.perf_counter()
       #start_time = time.clock()

        num_batches = ceil(len(train_data) / Config().batch_size)
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, device, shuffle=True, seed_value=seed_value)):
            uids, baskets, lens, prev_idx = x
            baskets = [[torch.tensor(basket, device=device) for basket in user] for user in baskets]
            model.zero_grad() 
            dynamic_user, _ = model(baskets, lens, dr_hidden)

            # loss = bpr_loss(uids, baskets, dynamic_user, model.encode.weight)
            #loss = bpr_loss(uids, baskets, dynamic_user, model.encode)
            loss = bpr_loss(uids, baskets, dynamic_user, model.encode, neg_samples, device=device)
            loss.backward()

            # Clip to avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config().clip)

            # Parameter updating
            optimizer.step()
            train_loss += loss.data

            # Logging
            #if i % Config().log_interval == 0 and i > 0:
            elapsed = (time.process_time() - start_time) / Config().log_interval
            cur_loss = train_loss.item() / Config().log_interval  # turn tensor into float
            train_loss = 0
            start_time = time.process_time()
            logger.info('[Training]| Epochs {:3d} | Batch {:5d} / {:5d} | ms/batch {:02.2f} | Loss {:05.4f} |'
                        .format(epoch, i+1, num_batches, elapsed, cur_loss))

    def validate_model():
        model.eval()  # Set the model to evaluation mode (no dropout)
        dr_hidden = model.init_hidden(Config().batch_size)  # Initialize hidden state
        dr_hidden = tuple(h.to(device) for h in dr_hidden) if isinstance(dr_hidden, tuple) else dr_hidden.to(device)
        val_loss = 0
        start_time = time.perf_counter()

        num_batches = ceil(len(validation_data) / Config().batch_size)
        with torch.no_grad():  # Disable gradient computation for validation
            for i, x in enumerate(dh.batch_iter(validation_data, Config().batch_size, Config().seq_len, device, shuffle=False, seed_value=seed_value)):
                uids, baskets, lens, prev_idx = x
                dynamic_user, _ = model(baskets, lens, dr_hidden)
                loss = bpr_loss(uids, baskets, dynamic_user, model.encode, neg_samples, device=device)
                val_loss += loss  # Accumulate GPU tensor directly

        # Average loss calculation
        val_loss = val_loss / num_batches
        elapsed = (time.perf_counter() - start_time) * 1000 / num_batches

        # Convert loss to float for logging
        logger.info('[Validation]| Epochs {:3d} | Elapsed {:02.2f} ms/batch | Loss {:05.4f} | '
                    .format(epoch, elapsed, val_loss.item()))
        return val_loss.item()
    
    # calculate recall 
    def recall_cal(positives, pred_items, count_at_least_one_cor_pred):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        if(correct_preds>=1): count_at_least_one_cor_pred += 1
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))
        return float(correct_preds/actual_bsize), count_at_least_one_cor_pred
    
    def test_model(offered_courses):
        model.eval()
        # item_embedding = model.encode.weight
        # item_embedding = model.encode
        # dr_hidden = model.init_hidden(Config().batch_size)
        item_embedding = model.encode.to(device)  # Move item embeddings to device
        dr_hidden = model.init_hidden(Config().batch_size)
        dr_hidden = tuple(h.to(device) for h in dr_hidden) if isinstance(dr_hidden, tuple) else dr_hidden.to(device)
        #dr_hidden = model.init_hidden(Config().batch_size)

        hitratio_numer = 0
        hitratio_denom = 0
        #ndcg = 0.0
        recall = 0.0
        recall_2= 0.0
        recall_temp = 0.0
        count=0
        count_at_least_one_cor_pred = 0
        #print(target_data)
        num_items = len(item_dict)
        list_of_top_items = {}
        for i, x in enumerate(dh.batch_iter(train_set_without_target, Config().batch_size, Config().seq_len, device, shuffle=False, seed_value=seed_value)):
            uids, baskets, lens, prev_idx = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            for uid, l, du in zip(uids, lens, dynamic_user):
                scores = []
                #user_baskets = train_set_without_target[train_set_without_target['userID'] == uid].baskets.values[0]
                #target_semester = target_data[target_data['userID'] == uid].last_semester.values[0]
                #du_latest = du[l - 1].unsqueeze(0)
                du_latest = du[l - 1].unsqueeze(0).to(device)
                user_baskets = train_set_without_target[train_set_without_target['userID'] == uid.item()].baskets.values[0]
                target_semester = target_data[target_data['userID'] == uid.item()].last_semester.values[0]
                positives = target_data[target_data['userID'] == uid.item()].baskets.values[0]  # list dim 1
                p_length = len(positives)
                item_list1 = [i for i in range(num_items)]
                #item_list_ten = torch.LongTensor(item_list1).to(device)
                item_list_ten = torch.tensor(item_list1).to(device)
                #scores = list(torch.mm(du_latest, item_embedding[item_list_ten].t()).data.numpy()[0])
                scores = torch.mm(du_latest, item_embedding[item_list_ten].t()).squeeze(0)
                
                #top_k1= Config().top_k
                top_k1 = len(positives)
                #print(offered_courses[l+1])
                k=0
                pred_items= []
                count1= 0
                
                _, sorted_indices = torch.sort(scores, descending=True)
                for index in sorted_indices:
                    #index = scores.index(max(scores))
                    item1 = item_list1[index]
                    if not utils.filtering(item1, user_baskets, offered_courses[target_semester], item_dict):
                        #if index not in index_k:
                        if item1 not in pred_items and len(pred_items)<top_k1:
                            #index_k.append(index)
                            pred_items.append(item1)
                            k+=1
                    #scores[index] = -9999
                    if len(pred_items)== top_k1: break
                    count1+= 1
                    if(count1==len(scores)): break
                
                hitratio_numer += len((set(positives) & set(pred_items)))
                hitratio_denom += p_length
                list_of_top_items[uid.item()] = pred_items
                
                recall_temp, count_at_least_one_cor_pred= recall_cal(positives, pred_items, count_at_least_one_cor_pred)
                recall_2+= recall_temp
                count=count+1
                
        hit_ratio = hitratio_numer / hitratio_denom
        #ndcg = ndcg / len(train_data)
        recall = recall_2/ count
        logger.info('[Test]| Epochs {:3d} | Hit ratio {:02.4f} | recall {:05.4f} |'
                    .format(epoch, hit_ratio, recall))
        print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
        percentage_of_at_least_one_cor_pred = count_at_least_one_cor_pred/ len(target_data)
        print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
        user_emb = {}
        #print(target_data)
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, device, shuffle=False, seed_value=seed_value)):
            uids, baskets, lens, prev_idx = x
            dynamic_user, _ = model(baskets, lens, dr_hidden)
            for uid, l, du in zip(uids, lens, dynamic_user):
                #scores = []
                #du_latest = du[l - 1].unsqueeze(0)
                du_latest = du[l - 1].unsqueeze(0).to(device)
                user_emb[uid.item()] = du_latest

        return hit_ratio, recall, user_emb, list_of_top_items

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_v20", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{hitratio:.4f}-{recall:.4f}.model'

    best_hit_ratio = None

    try:
        # Training
        for epoch in range(Config().epochs):
            train_model()
            logger.info('-' * 89)

            val_loss = validate_model()
            logger.info('-' * 89)

            hit_ratio, recall, user_emb, list_of_top_items = test_model(offered_courses)
            logger.info('-' * 89)

            # Checkpoint
            if not best_hit_ratio or hit_ratio > best_hit_ratio:
                with open(checkpoint_dir.format(epoch=epoch, hitratio=hit_ratio, recall=recall), 'wb') as f:
                    torch.save(model, f)
                best_hit_ratio = hit_ratio
    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')
    print("model directory: ", timestamp)
    print("config for train: 64, 2, 0.6")

    return timestamp, recall, user_emb, list_of_top_items

def recall_cal(positives, pred_items):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        return float(correct_preds/actual_bsize)
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))

# validation recall considering prereq connections
def valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, valid_data, valid_target, device):
    f = open(output_path, "w") #generating text file with recommendation using filtering function
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    #test_data = dh.load_data(Config().TRAININGSET_DIR)
    # valid_data = dh.load_data('./valid_sample_without_target.json')

    logger.info("✔︎ Test data processing...")
    #test_target = dh.load_data(Config().TESTSET_DIR)
    #valid_target = dh.load_data('./validation_target_set.json')

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    
    dr_model = torch.load(MODEL_DIR)
    dr_model = dr_model.to(device)  # Move to the appropriate devic

    dr_model.eval()

    # item_embedding = dr_model.encode.weight
    item_embedding = dr_model.encode
    hidden = dr_model.init_hidden(Config().batch_size)
    #dr_hidden = model.init_hidden(Config().batch_size)
    hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)

    hitratio_numer = 0
    hitratio_denom = 0
    #ndcg = 0.0
    recall = 0.0
    recall_2= 0.0
    #recall_3= 0.0
    count=0
    recall_bsize = {}
    num_items = len(item_dict)
            
    for i, x in enumerate(dh.batch_iter(valid_data, Config().batch_size, Config().seq_len, device, shuffle=False, seed_value=seed_value)):
        uids, baskets, lens, prev_idx = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            scores = []
            du_latest = du[l - 1].unsqueeze(0).to(device)
            user_baskets = valid_data[valid_data['userID'] == uid.item()].baskets.values[0]
            prior_bsize = len(user_baskets)
            #print("user_baskets: ", user_baskets)
            item_list1= []
            # calculating <u,p> score for all test items <u,p> pair
            positives = valid_target[valid_target['userID'] == uid.item()].baskets.values[0]  # list dim 1
            target_semester = valid_target[valid_target['userID'] == uid.item()].last_semester.values[0]

            p_length = len(positives)
            item_list1 = [i for i in range(num_items)]
            item_list_ten = torch.tensor(item_list1).to(device)
            scores = torch.mm(du_latest, item_embedding[item_list_ten].t()).squeeze(0)
            
            #top_k1= Config().top_k
            top_k1 = len(positives)
            recall_temp =0.0
            #print(offered_courses[l+1])
            if t_idx==1: # we are not considering randomly selected instances for last batch
                k=0
                pred_items= []
                count1= 0
                
                _, sorted_indices = torch.sort(scores, descending=True)
                for index in sorted_indices:
                    #index = scores.index(max(scores))
                    item1 = item_list1[index]
                    if not utils.filtering(item1, user_baskets, offered_courses[target_semester], item_dict):
                        #if index not in index_k:
                        if item1 not in pred_items and len(pred_items)< top_k1:
                            #index_k.append(index)
                            pred_items.append(item1)
                            k+=1
                    #scores[index] = -9999
                    if len(pred_items)== top_k1: break
                    count1+= 1
                    if(count1==len(scores)): break
                #print(index_k)
                #print(pred_items)
                f.write("UserID: ")
                # f.write(str(reversed_user_dict[reversed_user_dict2[uid]])+ "| ")
                f.write(str(reversed_user_dict2[uid.item()])+ "| ")
                #f.write(str(uid)+ "| ")
                f.write("target basket: ")
                target_basket2 = []
                for item2 in positives:
                    f.write(str(reversed_item_dict[item2])+ " ")
                    target_basket2.append(reversed_item_dict[item2])
                
                rec_basket2 = []
                f.write(", Recommended basket: ")
                for item3 in pred_items:
                    f.write(str(reversed_item_dict[item3])+ " ")
                    rec_basket2.append(reversed_item_dict[item3])
                f.write("\n") 
                    
                prior_courses = []
                for basket3 in user_baskets:
                    for item4 in basket3:
                        if reversed_item_dict[item4] not in prior_courses:
                            prior_courses.append(reversed_item_dict[item4])
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer += len((set(positives) & set(pred_items)))
                hitratio_denom += p_length
                #print(index_k)

                #calculate recall
                #recall_2+= recall_cal(positives, index_k)
                recall_temp = recall_cal(positives, pred_items)
                recall_2+= recall_temp
                if prior_bsize not in recall_bsize:
                    recall_bsize[prior_bsize]= [recall_temp]
                else:
                    recall_bsize[prior_bsize] += [recall_temp]
                
                count=count+1

    hitratio = hitratio_numer / hitratio_denom
    #ndcg = ndcg / len(test_data)
    recall = recall_2/ count
    print(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write("\n")
    #print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    print('Recall[@n]: {0}'.format(recall))
    f.write(str('Recall[@n]: {0}'.format(recall)))
    f.write("\n")
    
    f.close()
    return recall

def recall_cal_test(positives, len_target_basket, pred_items, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred):
        p_length= len_target_basket
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        if(correct_preds>=1): count_at_least_one_cor_pred += 1
        if correct_preds>=2: count_at_least_two_cor_pred+= 1
        if correct_preds>=3: count_at_least_three_cor_pred+= 1
        if correct_preds>=4: count_at_least_four_cor_pred+= 1
        if correct_preds>=5: count_at_least_five_cor_pred+= 1
        if correct_preds==actual_bsize: count_all_cor_pred+= 1

        if (actual_bsize>=6): 
            if(correct_preds==1): count_cor_pred[6,1]+= 1
            if(correct_preds==2): count_cor_pred[6,2]+= 1
            if(correct_preds==3): count_cor_pred[6,3]+= 1
            if(correct_preds==4): count_cor_pred[6,4]+= 1
            if(correct_preds==5): count_cor_pred[6,5]+= 1
            if(correct_preds>=6): count_cor_pred[6,6]+= 1
        else:
            if(correct_preds==1): count_cor_pred[actual_bsize,1]+= 1
            if(correct_preds==2): count_cor_pred[actual_bsize,2]+= 1
            if(correct_preds==3): count_cor_pred[actual_bsize,3]+= 1
            if(correct_preds==4): count_cor_pred[actual_bsize,4]+= 1
            if(correct_preds==5): count_cor_pred[actual_bsize,5]+= 1
        
        return float(correct_preds/actual_bsize), count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))
# testing with model 
def test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_data, test_target, device):
    f = open(output_path, "w") #generating text file with recommendation using filtering function
    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    dr_model = torch.load(MODEL_DIR)
    dr_model = dr_model.to(device)  # Move to the appropriate devic

    dr_model.eval()

    # item_embedding = dr_model.encode.weight
    item_embedding = dr_model.encode
    hidden = dr_model.init_hidden(Config().batch_size)
    #dr_hidden = model.init_hidden(Config().batch_size)
    hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)

    hitratio_numer = 0
    hitratio_denom = 0
    #ndcg = 0.0
    recall = 0.0
    recall_2= 0.0
    #recall_3= 0.0
    count=0
    count_at_least_one_cor_pred = 0
    total_correct_preds = 0
    recall_test_for_one_cor_pred = 0.0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
    recall_temp =0.0
    target_basket_size = {}
    target_basket_size[1] = 0
    target_basket_size[2] = 0
    target_basket_size[3] = 0
    target_basket_size[4] = 0
    target_basket_size[5] = 0
    target_basket_size[6] = 0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    
    recall_bsize = {}
    rec_info = []
   
    list_of_top_items = {}
    rank_list_of_top_items = {}
    user_emb_test = {}
    PHR_sum = 0
    num_items = len(item_dict)
    for i, x in enumerate(dh.batch_iter(test_data, Config().batch_size, Config().seq_len, device, shuffle=False, seed_value=seed_value)):
        uids, baskets, lens, prev_idx = x
        dynamic_user, _ = dr_model(baskets, lens, hidden)
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            scores = []
            du_latest = du[l - 1].unsqueeze(0).to(device)
            user_emb_test[uid.item()] = du_latest
            user_baskets = test_data[test_data['userID'] == uid.item()].baskets.values[0]
            prior_bsize = len(user_baskets)
            #print("user_baskets: ", user_baskets)
            item_list1= []
            # calculating <u,p> score for all test items <u,p> pair
            positives = test_target[test_target['userID'] == uid.item()].baskets.values[0]  # list dim 1
            target_semester = test_target[test_target['userID'] == uid.item()].last_semester.values[0]
            user_last_p_basket = user_baskets[-1]

            p_length = len(positives)
            item_list1 = [i for i in range(num_items)]
            item_list_ten = torch.tensor(item_list1).to(device)
            scores = torch.mm(du_latest, item_embedding[item_list_ten].t()).squeeze(0)
            
            top_k1 = len(positives)
            #print(offered_courses[l+1])
            if t_idx==1: # we are not considering randomly selected instances for last batch
                k=0
                pred_items= []
                count1= 0
                _, sorted_indices = torch.sort(scores, descending=True)
                rank_all_items = sorted_indices.tolist()
                #rank_all_items = []
                for index in sorted_indices:
                    #index = scores.index(max(scores))
                    item1 = item_list1[index]
                    if not utils.filtering(item1, user_baskets, offered_courses[target_semester], item_dict):
                        #if index not in index_k:
                        if item1 not in pred_items and len(pred_items)< top_k1:
                            #index_k.append(index)
                            pred_items.append(item1)
                            k+=1
                    # if item1 not in rank_all_items:
                    #     rank_all_items.append(item1)
                    #scores[index] = -9999
                    if len(pred_items)== top_k1: break
                    count1+= 1
                    if(count1==len(scores)): break
                
                f.write(str(reversed_user_dict3[uid.item()])+ "| ")
                f.write("target basket: ")
                target_basket2 = []
                for item2 in positives:
                    f.write(str(reversed_item_dict[item2])+ " ")
                    target_basket2.append(reversed_item_dict[item2])

                f.write(", Recommended basket: ")
                rec_basket2 = []
                for item3 in pred_items:
                    f.write(str(reversed_item_dict[item3])+ " ")
                    rec_basket2.append(reversed_item_dict[item3])
                prior_courses = []
                for basket3 in user_baskets:
                    for item4 in basket3:
                        if reversed_item_dict[item4] not in prior_courses:
                            prior_courses.append(reversed_item_dict[item4])

                f.write("\n") 
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer += len((set(positives) & set(pred_items)))
                hitratio_denom += p_length
                #print(index_k)
                pred_courses = []
                for item3 in pred_items:
                    pred_courses.append(reversed_item_dict[item3])
                list_of_top_items[uid.item()] = pred_items
               
                if count==0:
                    print("uid1 before: ", uid)
                    #print("uid_sample: ", uid_sample)
                    print("rec items before: ", pred_items)
                    print("target basket before: ", positives)
                rank_list_of_top_items[uid.item()] = rank_all_items
    
                #calculate recall
                recall_temp, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal_test(positives, top_k1, pred_items, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)  

                if recall_temp>0: PHR_sum += 1

                if top_k1>=2: count_actual_bsize_at_least_2 += 1
                if top_k1>=3: count_actual_bsize_at_least_3 += 1
                if top_k1>=4: count_actual_bsize_at_least_4 += 1
                if top_k1>=5: count_actual_bsize_at_least_5 += 1
                if top_k1>=6: count_actual_bsize_at_least_6 += 1
               
                rel_rec = len((set(positives) & set(pred_items)))
                row = [top_k1, target_basket2, pred_courses, rel_rec, recall_temp, target_semester]
                rec_info.append(row)
                
                if recall_temp>0:  
                    recall_test_for_one_cor_pred += recall_temp
                correct_preds2= len((set(positives) & set(pred_items)))
                total_correct_preds += correct_preds2
                if prior_bsize not in recall_bsize:
                    recall_bsize[prior_bsize]= [recall_temp]
                else:
                    recall_bsize[prior_bsize] += [recall_temp]
               
                if top_k1>=6: target_basket_size[6] += 1 
                else: target_basket_size[top_k1] += 1 
                recall_2+= recall_temp              
                count=count+1
            
    hitratio = hitratio_numer / hitratio_denom
    #ndcg = ndcg / len(test_data)
    print("total count: ", count)
    recall = recall_2/ count
    PHR = (PHR_sum/ count) * 100
    # print('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio))
    # f.write(str('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio)))
    print(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write(str('Hit ratio[@n]: {0}'.format(hitratio)))
    f.write("\n")
    #print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    print('Recall[@n]: {0}'.format(recall))
    f.write(str('Recall[@n]: {0}'.format(recall)))
    f.write("\n")
    print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
    f.write("count_at_least_one_cor_pred "+ str(count_at_least_one_cor_pred)+ "\n")
    percentage_of_at_least_one_cor_pred = (count_at_least_one_cor_pred/ len(test_target)) *100
    print("percentage_of_at_least_one_cor_pred: " + str(percentage_of_at_least_one_cor_pred)+"\n")
    f.write("percentage_of_at_least_one_cor_pred: " + str(percentage_of_at_least_one_cor_pred)+"\n")

    percentage_of_at_least_two_cor_pred = (count_at_least_two_cor_pred/ count_actual_bsize_at_least_2) *100
    print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
    f.write("percentage_of_at_least_two_cor_pred: "+ str(percentage_of_at_least_two_cor_pred)+ "\n")
    percentage_of_at_least_three_cor_pred = (count_at_least_three_cor_pred/ count_actual_bsize_at_least_3) *100
    print("percentage_of_at_least_three_cor_pred: ", percentage_of_at_least_three_cor_pred)
    f.write("percentage_of_at_least_three_cor_pred: "+ str(percentage_of_at_least_three_cor_pred)+ "\n")
    percentage_of_at_least_four_cor_pred = (count_at_least_four_cor_pred/ count_actual_bsize_at_least_4) * 100
    print("percentage_of_at_least_four_cor_pred: ", percentage_of_at_least_four_cor_pred)
    f.write("percentage_of_at_least_four_cor_pred: "+ str(percentage_of_at_least_four_cor_pred)+ "\n")
    percentage_of_at_least_five_cor_pred = (count_at_least_five_cor_pred/ count_actual_bsize_at_least_5) *100
    print("percentage_of_at_least_five_cor_pred: ", percentage_of_at_least_five_cor_pred)
    f.write("percentage_of_at_least_five_cor_pred: "+ str(percentage_of_at_least_five_cor_pred)+ "\n")
    percentage_of_all_cor_pred = (count_all_cor_pred/ len(test_target)) *100
    print("percentage_of_all_cor_pred: ", percentage_of_all_cor_pred)
    f.write("percentage_of_all_cor_pred: "+ str(percentage_of_all_cor_pred)+ "\n")
    
    f.write("\n") 
    f.close()
    return recall, percentage_of_at_least_one_cor_pred, percentage_of_at_least_two_cor_pred, list_of_top_items, user_emb_test, rank_list_of_top_items, PHR

if __name__ == '__main__':
    start = time.time()
    
    train_data_unique = pd.read_json('./train_data_all.json', orient='records', lines= True)
    train_data_all, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data_unique) 
    train_all, train_set_without_target, target_set, max_len = preprocess_train_data_part2(train_data_all) 
    valid_data = pd.read_json('./valid_data_all.json', orient='records', lines= True)

    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    #term_dict_train, frequency_of_courses_train, count_course_avg_train, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(train_data_unique, reversed_item_dict)
    #valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data_excluding_summer_term, reversed_user_dict, item_dict)
    valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data)
    test_data = pd.read_json('./test_data_all.json', orient='records', lines= True)
    test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
    test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data)

    negative_sample(Config().NEG_SAMPLES, train_data_all, valid_set_without_target, test_set_without_target)

    #offered_courses = calculate_offered_courses(train_all)
    offered_courses = offered_course_cal('./all_data.json')
    #train(offered_courses, train_set_without_target, reversed_item_dict, reversed_user_dict)

    dataTrain, dataTest, dataTotal, item_list, item_dict, reversed_item_dict, one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, reversed_item_dict_one_hot, user_dict_one_hot, one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train = preprocess_for_GAT.preprocess_data(train_data_unique)
    num_items = one_hot_encoded_train.shape[1]
    df = pd.read_json('./course_info_desc_keywords.json', orient='records', lines=True)
    one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx, reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level = convert_side_info_to_one_hot_encoding(df, reversed_item_dict_one_hot, num_items)
    window_s = 10
    #adj_mat, vocab_size = get_adj_matrix_text_v2(df, window_s, reversed_item_dict_one_hot)
    adj_mat, vocab_size, vocab_dict_idx_to_wrd, vocab_dict_wrd_to_idx = get_adj_matrix_text_v3(df, reversed_item_dict_one_hot, window_s)
    print(vocab_size)
    dense_adj_matrix_text = adj_mat.toarray()
    adj_matrix_text = normalize(dense_adj_matrix_text, norm='l2')
    #print(adj_matrix_text)
    threshold_weight_edges_iw = 0.1
    threshold_weight_edges_ww = 0.1
    
    n_layers = [1,2,3,4]
    embedding_dim = [32, 64, 128]
    n_epochs = [200, 300, 400]
    l_rate = [0.01, 0.001]
    edge_dropout = [0, 0.2]
    node_dropout = [0, 0.2]
    n_heads= [2,1,4]

    n_rnn_layers = [1, 2, 3]
    rnn_dropout = [0.3, 0.4, 0.5]
    rnn_l_rate = [0.01]
    num_users = one_hot_encoded_train.shape[0]
    num_items = one_hot_encoded_train.shape[1]
    num_cat_f = one_hot_encoded_cat.shape[1] 
    num_level_f = one_hot_encoded_level.shape[1]
    num_text_f = adj_matrix_text.shape[1]
    # num_fet = one_hot_encoded_cat.shape[1] + one_hot_encoded_level.shape[1]
    num_fet = num_cat_f + num_level_f + num_text_f
    device = torch.device("cpu")

    base_path = './saved_model_GAT_CDREAM_uis_text_GRU_avg_attn_layers_v3_keywords_tfidf/'
    all_results = []
    cnt3 = 0
    cnt4 = 1 # 0 
    for lr in l_rate:
        for n_lays in n_layers:
            for emb_dim in embedding_dim:
                for epoc in n_epochs:
                # for lr in l_rate:
                    for n_hds in n_heads:
                        for edge_drops in edge_dropout:
                            for node_drops in node_dropout:
                                model_filename = f"model_v{cnt4}.pth"
                                full_path = os.path.join(base_path, model_filename)
                                print("random number 1: ")
                                print(torch.rand(1))
                                model = GAT(num_users, num_items, num_fet, emb_dim, n_lays, edge_drops, node_drops, n_hds, seed_value).to(device)                             
                                if os.path.exists(full_path):
                                    print(f"Loading model from {full_path}")
                                    model.load_state_dict(torch.load(full_path))  # Load state_dict into the model
                                    item_embeddings = model.item_embeddings.weight.detach()
                                else:
                                    print(f"No saved model found at {full_path}. Training from scratch.")
                                    # model, data10, final_x, final_attn_weights, final_edge_index, kept_word_node_to_idx = train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, adj_matrix_text, cc_seq_matrix, n_lays, emb_dim, epoc, lr, edge_drops, node_drops, n_hds, threshold_weight_edges_iw, threshold_weight_edges_ww, threshold_weight_edges_cc, seed_value, cnt4)
                                    # item_embeddings = final_x[:num_items]
                                    model, data10, final_x, final_attn_weights, final_edge_index, kept_word_node_to_idx = train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, adj_matrix_text, n_lays, emb_dim, epoc, lr, edge_drops, node_drops, n_hds, threshold_weight_edges_iw, threshold_weight_edges_ww, seed_value, cnt4)
                                    item_embeddings = final_x[:num_items]
                                cnt4 += 1
                                #device = torch.device("cpu")
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                num_users_train = one_hot_encoded_train.shape[0]
                                item_embeddings_updated = update_idx_of_item_embedding(one_hot_encoded_train, num_users_train, item_embeddings, reversed_item_dict, item_dict, item_dict_one_hot) # reversed_item_dict = idx to cid
                                #item_embeddings_updated = torch.from_numpy(item_embeddings_updated).float().requires_grad_(True)
                                item_embeddings_updated = item_embeddings_updated.to(device)
                                
                                #device = torch.device("cpu")
                                for n_rnn_lays in n_rnn_layers:
                                    for rnn_drops in rnn_dropout:
                                        for rnn_lr in rnn_l_rate:
                                            MODEL_checkpoint, train_recall, user_emb_train, list_of_top_items_train = train(offered_courses, train_set_without_target, target_set, item_dict, item_embeddings_updated, train_data_all, valid_set_without_target, n_rnn_lays, rnn_drops, rnn_lr, device)
                                            MODEL_checkpoint = str(MODEL_checkpoint)
                                            end = time.time()
                                            total_training_time = end - start

                                            while not (MODEL_checkpoint.isdigit() and len(MODEL_checkpoint) == 10):
                                                MODEL_checkpoint = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
                                            logger.info("✔︎ The format of your input is legal, now loading to next step...")

                                            MODEL_DIR = dh.load_model_file(MODEL_checkpoint)
                                            data_dir= './'
                                            output_dir = data_dir + "/output_dir"
                                            utils.create_folder(output_dir)
                                            output_path= output_dir+ "/valid_prediction_28_v11001.txt"
                                            valid_recall = valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, valid_set_without_target, valid_target, device)
                                            output_path= output_dir+ "/test_prediction_28_v11001.txt"
                                            # term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, succ_swap_test, unsucc_swap_test, no_impact_swap_test, avg_rank_swap_test, std_rank_swap_test, cnt_swap_in_n_inst_test, cnt_tot_swap_test, avg_recall_swap_test, avg_recall_no_swap_test, tot_courses_test, cnt_succ_swaps_num_inst, cnt_unsucc_swaps_num_inst, cnt_no_impact_swaps_num_inst, cnt_both_succ_unsucc_swaps_num_inst, cnt_exact_1_swap_num_inst, cnt_exact_2_swaps_num_inst, cnt_at_least_3_swaps_num_inst, cnt_at_least_one_unsuc_and_at_least_two_swaps, cnt_at_least_two_unsuc_and_at_least_two_swaps = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, output_path, course_prereq)
                                            test_recall, percentage_of_at_least_one_cor_pred, percentage_of_at_least_two_cor_pred, list_of_top_items_test, user_emb_test, rank_list_of_top_items_test, PHR_test_main = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_set_without_target, test_target, device)
                                            print("test recall@n: ", test_recall)
                                            print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
                                            print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
                                            print("validation recall@n: ", valid_recall)
                                            # print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred2)
                                            print("train recall@n: ", train_recall)                                         
                                            print("n_layers: ", n_lays)
                                            print("emb_dim: ", emb_dim)
                                            print("epochs: ", epoc)
                                            print("learning rate: ", lr)
                                            print("n of heads:", n_hds)
                                            print("edge dropout: ", edge_drops)
                                            print("node dropout: ", node_drops)
                                            cnt3 += 1                                           
                                            print("cnt of iteration: ", cnt3)
                                            print( "num_of_rnn_layers: ", n_rnn_lays)
                                            print("dropout in rnn: ",  rnn_drops)
                                            print("lr in rnn: ",  rnn_lr)
                                            attention_weights = final_attn_weights[-1]  # Attention weights tensor
                                            edge_index = final_edge_index[-1]  # Edge index tensor                                         
                                            cat_dict = cat_dict_one_hot  # Category dictionary
                                            level_dict = level_dict_one_hot  # Level dictionary
                                            vocab_dict = vocab_dict_idx_to_wrd  # Vocabulary dictionary
                                            
                                            dr_model = torch.load(MODEL_DIR)
                                            dr_model = dr_model.to(device)  # Move to the appropriate devic
                                            dr_model.eval()
                                            # item_embedding = dr_model.encode.weight
                                            item_embedding_all = dr_model.encode
                                            #explanation_paths = []
                                            explanation_paths_dict = {}
                                            explanation_paths_dict_collab = {}
                                            explanation_sim_dict = {}
                                            #cnt10= 0
                                            #uid1= 0
                                            test_users = list(test_set_without_target['userID'])  # List of 1000 users (e.g., user IDs from 1 to 1000)
                                            # Take a random sample of 100 users
                                            sampled_users = random.sample(test_users, 1259)                                           
                                            test_set_without_target_v2 = pd.concat([test_set_without_target], ignore_index=True).copy()
                                            #test_set_without_target_v2["baskets"] = test_set_without_target_v2["baskets"].apply(lambda x: [y for y in x])
                                            test_set_without_target_v2["baskets"] = test_set_without_target_v2["baskets"].apply(lambda x: [y[:] for y in x])
                                            #test_set_without_target_v2["baskets"] = test_set_without_target_v2["baskets"].apply(lambda x: list(x))
                                            test_set_without_target_v3 = pd.concat([test_set_without_target], ignore_index=True).copy()
                                            #test_set_without_target_v2["baskets"] = test_set_without_target_v2["baskets"].apply(lambda x: [y for y in x])
                                            test_set_without_target_v3["baskets"] = test_set_without_target_v3["baskets"].apply(lambda x: [y[:] for y in x])
                                            original_rec_items = []
                                           
                                            cnt10 = 0
                                            cnt_same_items_removed = 0
                                            cnt_same_items_removed_50 = 0
                                            sampled_users_rel_rec = []
                                            ground_truth_items = []
                                            rank_explained_rec_item_original = {}  #perturbed rank- original rank, compute avg rank of the rec item, compute NDCG for explained rec item
                                            recall_sum2 = 0
                                            PHR2_sum = 0
                                            cnt33= 0
                                            n_sample_with_rel_rec = 50
                                            list_of_rec_items = []
                                            for cnt_sample_users in range(len(sampled_users)):
                                                #item1 = items[0] # first item recommended for that test user
                                                uid_sample = sampled_users[cnt_sample_users] # actual user id
                                            # for cnt_sample_users in range(len(test_users)):
                                            #     uid_sample = test_users[cnt_sample_users]
                                                target_basket = test_target[test_target['userID'] == uid_sample].baskets.values[0]
                                                #uid1 = uid_sample - num_users_train - len(valid_set_without_target)  # userid starting from 0 to match list_of_top_items_test

                                                items = list_of_top_items_test[uid_sample] # rec items
                                                if cnt_sample_users==0:
                                                    #print("uid1: ", uid1)
                                                    print("uid_sample: ", uid_sample)
                                                    print("rec items: ", items)
                                                    print("target basket: ", target_basket)
                                                rel_rec = len(set(items) & set(target_basket))
                                                recall_sum2 += (rel_rec/len(target_basket))
                                                if rel_rec>0: PHR2_sum += 1
                                                cnt33+= 1
                                                item1 = None
                                                for rank1, item3 in enumerate(items, start=1):  
                                                    if item3 in target_basket:  # Correctly recommended item
                                                        item1 = item3
                                                        
                                                        break  

                                                if item1 is None:  
                                                    continue  # Skip if no correct item is found
                                                if item1 in list_of_rec_items: # and cnt_sample_users<1100:
                                                    continue

                                                if item1 not in list_of_rec_items: list_of_rec_items.append(item1)
                                                cid = reversed_item_dict[item1] # idx to cid in training data
                                                rec_item_node = reversed_item_dict_one_hot[cid] # cid to idx in GNN model                                              
                                                user_baskets = list(test_set_without_target[test_set_without_target['userID'] == uid_sample].baskets.values[0])                                          
                                                user_last_p_basket = user_baskets[-1] # last prior basket items
                                                optimal_path_nodes = []
                                                optimal_path_main = []
                                                optimal_path_nodes_collab = []
                                                optimal_path_main_collab = []
                                                min_path_length = 1000000000
                                                max_score = -10000000
                                                min_path_length2 = 1000000000
                                                max_score2 = -10000000
                                                most_similar_item= []
                                                most_similar_item_nodes= []
                                                #user_all_prior_items = [item for bskt in user_baskets for item in bskt]
                                                prior_cnames = []
                                                for p_item in user_last_p_basket:
                                                #for p_item in user_all_prior_items:
                                                    cid2 = reversed_item_dict[p_item] # idx to cid in test data
                                                    p_item_node = reversed_item_dict_one_hot[cid2] # cid to idx in GNN model (edges)

                                                    optimal_path, path_main, path_length = path_generation(
                                                        attention_weights, edge_index, item_dict_one_hot, user_dict_one_hot, cat_dict, 
                                                        level_dict, vocab_dict, kept_word_node_to_idx, num_items, 
                                                        num_users, num_cat_f, num_level_f, num_text_f, df, reversed_item_dict_one_hot, s1=p_item_node, d1=rec_item_node
                                                    )
                                                    optimal_path2, path_main2, path_length2 = path_generation_collab(
                                                        attention_weights, edge_index, item_dict_one_hot, user_dict_one_hot, cat_dict, 
                                                        level_dict, vocab_dict, kept_word_node_to_idx, num_items, 
                                                        num_users, num_cat_f, num_level_f, num_text_f, df, reversed_item_dict_one_hot, s1=p_item_node, d1=rec_item_node
                                                    )
                                                    if cnt_sample_users==0 or cnt_sample_users==1: 
                                                        print(f"Optimal path for {uid_sample} from {cid2} to {cid} is {path_main}")

                                                    if cnt_sample_users==0 or cnt_sample_users==1: 
                                                        print(f"Collaborative Optimal path for {uid_sample} from {cid2} to {cid} is {path_main2}")

                                                    if path_length < min_path_length:
                                                        min_path_length= path_length # lowest distance, highest attention weights
                                                        optimal_path_nodes = optimal_path 
                                                        optimal_path_main = path_main
                                                    if path_length2 < min_path_length2:
                                                        min_path_length2= path_length2 # lowest distance, highest attention weights
                                                        optimal_path_nodes_collab = optimal_path2
                                                        optimal_path_main_collab = path_main2
                                                    # computing cosine similarity
                                                    score_item = cal_similarity_score(p_item, item1, item_embedding_all)
                                                    if cnt_sample_users==0 or cnt_sample_users==1: 
                                                        print(f"cosine sim score for {uid_sample} from {cid2} to {cid} is {score_item}")
                                                    
                                                    cid_prior1 = reversed_item_dict[p_item]
                                                    cid_rec1 = reversed_item_dict[item1]
                                                    idxp = df[df['course_code']==cid_prior1].index.values[0]
                                                    cname1 = df["course_name"][idxp]
                                                    prior_cnames.append(cname1)

                                                    if score_item > max_score:
                                                        max_score= score_item # lowest distance, highest attention weights
                                                        #optimal_path_nodes = optimal_path 
                                                        
                                                        # index1 = dataTest_prev[dataTest_prev['userID'] == uid_sample].index.values[0]
                                                        
                                                        idxr = df[df['course_code']==cid_rec1].index.values[0]
                                                        # index1 = dataTest_prev[dataTest_prev['userID'] == uid_sample].index.values[0]
                                                        cname2 = df["course_name"][idxr]
                                                        # most_similar_item = [reversed_item_dict[p_item], reversed_item_dict[item1], score_item] # 
                                                        most_similar_item = [cname1, cname2] # 
                                                        most_similar_item_nodes = [p_item, item1]
                                                
                                                #explanation_paths.extend(optimal_path_main)
                                                if len(optimal_path_main)>0:
                                                    explanation_paths_dict[uid_sample] = [prior_cnames, optimal_path_main] # Save the path for the current user
                                                    explanation_paths_dict_collab[uid_sample] = [prior_cnames, optimal_path_main_collab] # Save the path for the current user
                                                    explanation_sim_dict[uid_sample] = [prior_cnames, most_similar_item]
                                                    sampled_users_rel_rec.append(uid_sample)
                                                    ground_truth_items.append(target_basket)
                                                    original_rec_items.append(items)
                                                    rank_explained_rec_item_original[uid_sample] = [item1, rank1]
                                                    # for one recommended item
                                                    #p_important_course = optimal_path_main[0] # important prior course
                                                    p_important_course_node = optimal_path_nodes[0] # important prior course node
                                                    cid3 = item_dict_one_hot[p_important_course_node] # node idx to cid in LightGCN/ GAT
                                                   
                                                    p_important_course = item_dict[cid3] # cid to idx in test data
                                                    if cnt10==0 or cnt10==1:
                                                        print("cidx: ", p_important_course)
                                                        print("c in prior: ", cid3)
                                                        print("user_last_p_basket: ", user_last_p_basket)
                                                        print("item1: ", item1)
                                                        print("uid: ", uid_sample)
                                                    user_last_p_basket.remove(p_important_course) # removing an item (user-item connection)
                                                    if len(user_last_p_basket)>0:
                                                        updated_baskets = user_baskets[:-1]+ [user_last_p_basket]
                                                    else:
                                                        updated_baskets = user_baskets[:-1]

                                                    if cnt10==0 or cnt10==1:
                                                        print("user_last_p_basket after removing a prior item: ", user_last_p_basket)
                                                        print("user baskets before: ", user_baskets)
                                                        print("user baskets after removing a prior item: ", updated_baskets)
                                                        
                                                    
                                                    #test_set_without_target['baskets'] = test_set_without_target['baskets'].astype(object)
                                                    # Update the DataFrame with the modified baskets
                                                    index1 = test_set_without_target[test_set_without_target['userID'] == uid_sample].index.values[0]
                                                    
                                                    # Now assign the new list back
                                                    
                                                    test_set_without_target.at[index1, 'baskets'] = updated_baskets
                                                    test_set_without_target.at[index1, 'num_baskets'] = len(updated_baskets)
                                                    
                                                    # for most similar item
                                                    p_important_course_2 = most_similar_item_nodes[0] # important prior course idx in test data
                                                    if p_important_course==p_important_course_2: cnt_same_items_removed += 1
                                                    if p_important_course==p_important_course_2 and cnt10<50: cnt_same_items_removed_50 += 1
                                                    
                                                    user_baskets2 = list(test_set_without_target_v2[test_set_without_target_v2['userID'] == uid_sample].baskets.values[0])
                                                    #t_basket = test_target[test_target['userID'] == uid_sample].baskets.values[0]
                                                    user_last_p_basket2 = user_baskets2[-1] # last prior basket items
                                                    # #print("user_last_p_basket2: ", user_last_p_basket2)
                                                    # #print("cidx3: ", cidx3)
                                                    user_last_p_basket2.remove(p_important_course_2) # removing an item (user-item connection)
                                                   
                                                    if len(user_last_p_basket2)>0:
                                                        updated_baskets2 = user_baskets2[:-1]+ [user_last_p_basket2]
                                                    else:
                                                        updated_baskets2 = user_baskets2[:-1]
                                                
                                                    if cnt10==0 or cnt10==1:
                                                        print("cidx3 for sim: ", p_important_course_2)
                                                        print("user baskets before sim: ", user_baskets2)
                                                        print("user_last_p_basket after removing a prior item based on similarity: ", user_last_p_basket2)
                                                        #print("user_last_p_basket2: ", user_last_p_basket2)
                                                        print("user baskets after removing a prior item based on similarity: ", updated_baskets2)
                                                    
                                                    # Update the DataFrame with the modified baskets
                                                    index22 = test_set_without_target_v2[test_set_without_target_v2['userID'] == uid_sample].index.values[0]
                                                   
                                                    test_set_without_target_v2.at[index22, 'baskets'] = updated_baskets2
                                                    test_set_without_target_v2.at[index22, 'num_baskets'] = len(updated_baskets2)
                                                    
                                                    p_important_course_node3 = optimal_path_nodes_collab[0] # important prior course node
                                                    cid4 = item_dict_one_hot[p_important_course_node3] # node idx to cid in LightGCN/ GAT
                                                    p_important_course3 = item_dict[cid4] # cid to idx in test data
                                                    user_baskets3 = list(test_set_without_target_v3[test_set_without_target_v3['userID'] == uid_sample].baskets.values[0])
                                                    #t_basket = test_target[test_target['userID'] == uid_sample].baskets.values[0]
                                                    user_last_p_basket3 = user_baskets3[-1] # last prior basket items
                                                    if cnt10==0 or cnt10==1:
                                                        print("cidx: ", p_important_course3)
                                                        print("c in prior: ", cid4)
                                                        print("user_last_p_basket collab: ", user_last_p_basket3)
                                                        print("item1: ", item1)
                                                        print("uid: ", uid_sample)
                                                    user_last_p_basket3.remove(p_important_course3) # removing an item (user-item connection)
                                                    if len(user_last_p_basket3)>0:
                                                        updated_baskets3 = user_baskets3[:-1]+ [user_last_p_basket3]
                                                    else:
                                                        updated_baskets3 = user_baskets3[:-1]

                                                    if cnt10==0 or cnt10==1:
                                                        print("user_last_p_basket after removing a prior item collab explanation: ", user_last_p_basket3)
                                                        print("user baskets before: ", user_baskets3)
                                                        print("user baskets after removing a prior item: ", updated_baskets3)
                                                    
                                                    #test_set_without_target['baskets'] = test_set_without_target['baskets'].astype(object)
                                                    # Update the DataFrame with the modified baskets
                                                    index23 = test_set_without_target_v3[test_set_without_target_v3['userID'] == uid_sample].index.values[0]
                                                    # Now assign the new list back
                                                    
                                                    test_set_without_target_v3.at[index23, 'baskets'] = updated_baskets3
                                                    test_set_without_target_v3.at[index23, 'num_baskets'] = len(updated_baskets3)

                                                    cnt10 += 1
                                                    if cnt10==n_sample_with_rel_rec: break
                                                else: continue
                                                    
                                            recall_test_main2 = recall_sum2/cnt33
                                            PHR2_test = PHR2_sum/cnt33
                                            test_recall_perturbed, percentage_of_at_least_one_cor_pred_perturbed, percentage_of_at_least_two_cor_pred_perturbed, list_of_top_items_test_perturbed, user_emb_test_perturbed, rank_list_of_top_items_test_perturbed, PHR_test_perturbed = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_set_without_target, test_target, device)
                                            # list_of_top_items_test_perturbed
                                            print("test recall perturbed: ", test_recall_perturbed)
                                            #cnt_sample_users = 0
                                            perturbed_rec_items = []
                                            # ground_truth_items = []
                                            rank_explained_rec_item_perturbed = {}
                                            changed_rec_cnt = 0
                                            changed_rec_cnt_50 = 0
                                            for cnt_sample_users in range(len(sampled_users_rel_rec)):
                                                #item1 = items[0] # first item recommended for that test user
                                                uid_sample = sampled_users_rel_rec[cnt_sample_users] # actual user id
                                                #uid1 = uid_sample - num_users_train - len(valid_set_without_target)  # userid starting from 0 to match list_of_top_items_test
                                                items = list_of_top_items_test_perturbed[uid_sample] # rec items perturbing graph (user-item connections)
                                                rank_all_items_v2 = rank_list_of_top_items_test_perturbed[uid_sample]
                                                perturbed_rec_items.append(items)
                                                user_baskets = test_set_without_target[test_set_without_target['userID'] == uid_sample].baskets.values[0]
                                                user_last_p_basket = user_baskets[-1] # last prior basket items
    
                                                item1, rank_original = rank_explained_rec_item_original[uid_sample]
                                                if cnt_sample_users==0 or cnt_sample_users==1:
                                                    print("uid: ", uid_sample)
                                                    print("item1: ", item1)
                                                if item1 not in items:
                                                    changed_rec_cnt += 1
                                                    try:
                                                        index_item1 = rank_all_items_v2.index(item1)
                                                        print("changed ranking: ", index_item1)
                                                        #print(f"Item found at index: {index}")
                                                    except ValueError:
                                                        index_item1 = len(items)+10 # impossible case (if item not found)
                                                        print("uid: ", uid_sample)
                                                        print("len of ranking of all items sorted: ", len(rank_all_items_v2))
                                                        print("rank of all items: ", rank_all_items_v2)
                                                        print(f"Item {item1} not found in the list.")
                                                        print("length of user's last prior basket", len(user_last_p_basket))
                                                    rank_explained_rec_item_perturbed[uid_sample] = [item1, index_item1+1] # rank + 1
                                                else:
                                                    index_item1 = items.index(item1)
                                                    rank_explained_rec_item_perturbed[uid_sample] = [item1, index_item1+1] # rank starts from 1
                                                if item1 not in items and cnt_sample_users<50:
                                                    changed_rec_cnt_50 += 1

                                            fidelity_plus, original_recall, perturbed_recall, fidelity_rank, avg_rank_org, avg_rank_per, ndcg_original_explained_item, ndcg_perturbed_explained_item, ndcg_original_all_rec_items, ndcg_perturbed_all_rec_items = fidelity_plus_measure_main_v2(ground_truth_items, original_rec_items, perturbed_rec_items, rank_explained_rec_item_original, rank_explained_rec_item_perturbed)
                                            for k4,v4 in explanation_paths_dict.items():
                                                dict3 = {}
                                                dict3["prior courses"] = v4[0]
                                                dict3["explanation path"] = v4[1]
                                                explanation_paths_dict[k4] = dict3
                                            for k5,v5 in explanation_paths_dict_collab.items():
                                                dict4 = {}
                                                dict4["prior courses"] = v5[0]
                                                dict4["explanation path"] = v5[1]
                                                explanation_paths_dict_collab[k5] = dict4
                                            for k6,v6 in explanation_sim_dict.items():
                                                dict5 = {}
                                                dict5["prior courses"] = v6[0]
                                                dict5["similar item"] = v6[1]
                                                explanation_sim_dict[k6] = dict5
                                            with open("./explanation_paths_GAT_GRU_uist_50_sample9_v1_last_semester_items.json", "w") as json_file:
                                                json.dump(explanation_paths_dict, json_file, indent=4)
                                            with open("./explanation_paths_collab_GAT_GRU_uist_50_sample9_v1_last_semester_items.json", "w") as json_file:
                                                json.dump(explanation_paths_dict_collab, json_file, indent=4)
                                            with open("./explanation_sim_GAT_GRU_uist_50_sample9_v1_last_semester_items.json", "w") as json_file:
                                                json.dump(explanation_sim_dict, json_file, indent=4)

                                            # print("Explanation paths saved to explanation_paths_v9_200_samples.json")
                                            # print("Explanation paths based on similarity saved to explanation_sim_v8.json")
                                            per_inst_changed_rec = (changed_rec_cnt/len(sampled_users_rel_rec))*100
                                            per_inst_changed_rec_50 = changed_rec_cnt_50
                                            print("percentage of sample instance of changed_rec: ", per_inst_changed_rec)
                                            print("ground truth items length: ",len(ground_truth_items))
                                            print("original rec items length: ",len(original_rec_items))
                                            print("perturbed rec items length: ",len(perturbed_rec_items))

                                            test_recall_perturbed_sim, percentage_of_at_least_one_cor_pred_perturbed_sim, percentage_of_at_least_two_cor_pred_perturbed_sim, list_of_top_items_test_perturbed_sim, user_emb_test_perturbed_sim, rank_list_of_top_items_test_perturbed_sim, PHR_test_perturbed_sim = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_set_without_target_v2, test_target, device)
                                            # list_of_top_items_test_perturbed
                                            print("test recall perturbed sim: ", test_recall_perturbed_sim)
                                            #cnt_sample_users = 0
                                            perturbed_rec_items_sim = []
                                            # ground_truth_items = []
                                            rank_explained_rec_item_perturbed_sim = {}
                                            changed_rec_cnt_sim = 0
                                            changed_rec_cnt_sim_50 = 0
                                            for cnt_sample_users in range(len(sampled_users_rel_rec)):
                                                #item1 = items[0] # first item recommended for that test user
                                                uid_sample = sampled_users_rel_rec[cnt_sample_users] # actual user id
                                                #uid1 = uid_sample - num_users_train - len(valid_set_without_target)  # userid starting from 0 to match list_of_top_items_test
                                                items = list_of_top_items_test_perturbed_sim[uid_sample] # rec items perturbing graph (user-item connections)
                                                rank_all_items_v2_sim = rank_list_of_top_items_test_perturbed_sim[uid_sample]
                                                perturbed_rec_items_sim.append(items)
                                                user_baskets2 = test_set_without_target_v2[test_set_without_target_v2['userID'] == uid_sample].baskets.values[0]
                                                user_last_p_basket2 = user_baskets2[-1] # last prior basket items
                                        
                                                
                                                #ground_truth_items.append(t_basket)
                                                item1, rank_original = rank_explained_rec_item_original[uid_sample]
                                                if cnt_sample_users==0 or cnt_sample_users==1:
                                                    print("uid: ", uid_sample)
                                                    print("item1: ", item1)
                                                if item1 not in items:
                                                    changed_rec_cnt_sim += 1
                                                    try:
                                                        index_item1 = rank_all_items_v2_sim.index(item1)
                                                        print("changed ranking: ", index_item1)
                                                        #print(f"Item found at index: {index}")
                                                    except ValueError:
                                                        index_item1 = len(items)+10 # impossible case (if item not found)
                                                        print("impossible case")
                                                        print("uid: ", uid_sample)
                                                        print("len of ranking of all items sorted _sim: ", len(rank_all_items_v2_sim))
                                                        print("rank of all items _sim: ", rank_all_items_v2_sim)
                                                        print(f"Item {item1} not found in the list.")
                                                        print("length of user's last prior basket _sim", len(user_last_p_basket2))
                                                    rank_explained_rec_item_perturbed_sim[uid_sample] = [item1, index_item1+1] # rank + 1
                                                else:
                                                    index_item1 = items.index(item1)
                                                    rank_explained_rec_item_perturbed_sim[uid_sample] = [item1, index_item1+1] # rank starts from 1
                                                if item1 not in items and cnt_sample_users<50:
                                                    changed_rec_cnt_sim_50 += 1

                                            fidelity_plus_sim, original_recall_sim, perturbed_recall_sim, fidelity_rank_sim, avg_rank_org_sim, avg_rank_per_sim, ndcg_original_explained_item_sim, ndcg_perturbed_explained_item_sim, ndcg_original_all_rec_items_sim, ndcg_perturbed_all_rec_items_sim = fidelity_plus_measure_main_v2(ground_truth_items, original_rec_items, perturbed_rec_items_sim, rank_explained_rec_item_original, rank_explained_rec_item_perturbed_sim)
                                            # with open("./explanation_paths_v11_sim.json", "w") as json_file:
                                            #     json.dump(explanation_paths_dict, json_file, indent=4)

                                            # print("Explanation paths saved to explanation_paths_v11.json")
                                            per_inst_changed_rec_sim = (changed_rec_cnt_sim/len(sampled_users_rel_rec))*100
                                            per_inst_changed_rec_sim_50 = changed_rec_cnt_sim_50
                                            print("percentage of sample instance of changed_rec: ", per_inst_changed_rec_sim)
                                            print("ground truth items length: ",len(ground_truth_items))
                                            print("original rec items length: ",len(original_rec_items))
                                            print("perturbed rec items length: ",len(perturbed_rec_items_sim))
                                            print("PHR test main: ", PHR_test_main)
                                            print("recall test main from loop: ", recall_test_main2)
                                            print("PHR test main from loop: ", PHR2_test)

                                            test_recall_perturbed_col, percentage_of_at_least_one_cor_pred_perturbed_col, percentage_of_at_least_two_cor_pred_perturbed_col, list_of_top_items_test_perturbed_col, user_emb_test_perturbed_col, rank_list_of_top_items_test_perturbed_col, PHR_test_perturbed_col = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_set_without_target_v3, test_target, device)
                                            # list_of_top_items_test_perturbed
                                            print("test recall perturbed col: ", test_recall_perturbed_col)
                                            #cnt_sample_users = 0
                                            perturbed_rec_items_col = []
                                            # ground_truth_items = []
                                            rank_explained_rec_item_perturbed_col = {}
                                            changed_rec_cnt_col = 0
                                            changed_rec_cnt_col_50 = 0
                                            for cnt_sample_users in range(len(sampled_users_rel_rec)):
                                                #item1 = items[0] # first item recommended for that test user
                                                uid_sample = sampled_users_rel_rec[cnt_sample_users] # actual user id
                                                #uid1 = uid_sample - num_users_train - len(valid_set_without_target)  # userid starting from 0 to match list_of_top_items_test
                                                items = list_of_top_items_test_perturbed_col[uid_sample] # rec items perturbing graph (user-item connections)
                                                rank_all_items_v2_col = rank_list_of_top_items_test_perturbed_col[uid_sample]
                                                perturbed_rec_items_col.append(items)
                                                user_baskets3 = test_set_without_target_v3[test_set_without_target_v3['userID'] == uid_sample].baskets.values[0]
                                                user_last_p_basket3 = user_baskets3[-1] # last prior basket items
                                        
                                                
                                                #ground_truth_items.append(t_basket)
                                                item1, rank_original = rank_explained_rec_item_original[uid_sample]
                                                if cnt_sample_users==0 or cnt_sample_users==1:
                                                    print("uid: ", uid_sample)
                                                    print("item1: ", item1)
                                                if item1 not in items:
                                                    changed_rec_cnt_col += 1
                                                    try:
                                                        index_item1 = rank_all_items_v2_col.index(item1)
                                                        print("changed ranking: ", index_item1)
                                                        #print(f"Item found at index: {index}")
                                                    except ValueError:
                                                        index_item1 = len(items)+10 # impossible case (if item not found)
                                                        print("impossible case")
                                                        print("uid: ", uid_sample)
                                                        print("len of ranking of all items sorted _col: ", len(rank_all_items_v2_col))
                                                        print("rank of all items_col: ", rank_all_items_v2_col)
                                                        print(f"Item {item1} not found in the list.")
                                                        print("length of user's last prior basket_col", len(user_last_p_basket3))
                                                    rank_explained_rec_item_perturbed_col[uid_sample] = [item1, index_item1+1] # rank + 1
                                                else:
                                                    index_item1 = items.index(item1)
                                                    rank_explained_rec_item_perturbed_col[uid_sample] = [item1, index_item1+1] # rank starts from 1
                                                if item1 not in items and cnt_sample_users<50:
                                                    changed_rec_cnt_col_50 += 1

                                            fidelity_plus_col, original_recall_col, perturbed_recall_col, fidelity_rank_col, avg_rank_org_col, avg_rank_per_col, ndcg_original_explained_item_col, ndcg_perturbed_explained_item_col, ndcg_original_all_rec_items_col, ndcg_perturbed_all_rec_items_col = fidelity_plus_measure_main_v2(ground_truth_items, original_rec_items, perturbed_rec_items_col, rank_explained_rec_item_original, rank_explained_rec_item_perturbed_col)
                                            # with open("./explanation_paths_v11_col.json", "w") as json_file:
                                            #     json.dump(explanation_paths_dict, json_file, indent=4)

                                            # print("Explanation paths saved to explanation_paths_v11.json")
                                            per_inst_changed_rec_col = (changed_rec_cnt_col/len(sampled_users_rel_rec))*100
                                            per_inst_changed_rec_col_50 = changed_rec_cnt_col_50
                                            print("percentage of sample instance of changed_rec: ", per_inst_changed_rec_col)
                                            print("ground truth items length: ",len(ground_truth_items))
                                            print("original rec items length: ",len(original_rec_items))
                                            print("perturbed rec items length: ",len(perturbed_rec_items_col))
                                            print("PHR test main: ", PHR_test_main)
                                            print("recall test main from loop: ", recall_test_main2)
                                            print("PHR test main from loop: ", PHR2_test)
                                            print("num of unique items recommended: ", len(list_of_rec_items))
                                            row1 = [n_sample_with_rel_rec, n_lays, emb_dim, epoc, lr, n_hds, edge_drops, node_drops, threshold_weight_edges_iw, n_rnn_lays, rnn_drops, rnn_lr, train_recall, valid_recall, test_recall, percentage_of_at_least_one_cor_pred, percentage_of_at_least_two_cor_pred, test_recall_perturbed, fidelity_plus, original_recall, perturbed_recall, per_inst_changed_rec, fidelity_rank, avg_rank_org, avg_rank_per, ndcg_original_explained_item, ndcg_perturbed_explained_item, ndcg_original_all_rec_items, ndcg_perturbed_all_rec_items, test_recall_perturbed_sim, fidelity_plus_sim, original_recall_sim, perturbed_recall_sim, per_inst_changed_rec_sim, fidelity_rank_sim, avg_rank_org_sim, avg_rank_per_sim, ndcg_original_explained_item_sim, ndcg_perturbed_explained_item_sim, ndcg_original_all_rec_items_sim, ndcg_perturbed_all_rec_items_sim, test_recall_perturbed_col, fidelity_plus_col, original_recall_col, perturbed_recall_col, per_inst_changed_rec_col, fidelity_rank_col, avg_rank_org_col, avg_rank_per_col, ndcg_original_explained_item_col, ndcg_perturbed_explained_item_col, ndcg_original_all_rec_items_col, ndcg_perturbed_all_rec_items_col,cnt_same_items_removed]
                                            all_results.append(row1)
                                            all_results_df = pd.DataFrame(all_results, columns=['n_sample', 'n_layers', 'emb_dim', 'n_epochs', 'learning_rate', 'n_heads', 'edge_dropout', 'node_dropout', 'threshold_weight_edges_iw', 'rnn_layer_number', 'rnn_dropout', 'rnn_l_rate', 'train_recall', 'valid_recall', 'test_recall', 'percentage_of_at_least_one_cor_pred', 'percentage_of_at_least_two_cor_pred', 'test_recall_perturbed', 'fidelity_plus_recall', 'original_recall', 'perturbed_recall', 'per_inst_changed_rec', 'fidelity_ranking_explained_item', 'avg_rank_original', 'avg_rank_perturbed', 'ndcg_original_explained_item', 'ndcg_perturbed_explained_item', 'ndcg_original_all_rec_items', 'ndcg_perturbed_all_rec_items', 'test_recall_perturbed_sim', 'fidelity_plus_sim', 'original_recall_sim', 'perturbed_recall_sim', 'per_inst_changed_rec_sim', 'fidelity_rank_sim', 'avg_rank_org_sim', 'avg_rank_per_sim', 'ndcg_original_explained_item_sim', 'ndcg_perturbed_explained_item_sim', 'ndcg_original_all_rec_items_sim', 'ndcg_perturbed_all_rec_items_sim', 'test_recall_perturbed_col', 'fidelity_plus_col', 'original_recall_col', 'perturbed_recall_col', 'per_inst_changed_rec_col', 'fidelity_rank_col', 'avg_rank_org_col', 'avg_rank_per_col', 'ndcg_original_explained_item_col', 'ndcg_perturbed_explained_item_col', 'ndcg_original_all_rec_items_col', 'ndcg_perturbed_all_rec_items_col', 'cnt_same_items_removed_cfeat_path_sim'])
                                            
                                            all_results_df.to_json('./all_results_df_LLM_GAT_GRU_uist_keywords_tfidf_ckw_explain_50_sample9_v1_last_semester_items.json', orient='records', lines=True) 
                                            all_results_df.to_csv('./all_results_df_LLM_GAT_GRU_uist_keywords_tfidf_ckw_explain_50_sample9_v1_last_semester_items.csv') 
                                            
