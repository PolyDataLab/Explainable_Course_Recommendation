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
#from utils import data_helpers as dh
import data_helpers_v3_cr_genrec_ser as dh
from config_v8_cr_genrec_ser import Config # type: ignore
from genrec_model_v2_cr import GenRec
import tensorflow as tf
from dataprocess_cr_genrec_ser_v2 import *
#from utils import *
import utils_CDREAM # type: ignore
from utils_CDREAM import * # type: ignore
#from offered_courses import *
import pandas as pd
#import preprocess_v30_tafeng_ser
#from offered_courses_v2 import *
#from topic_model_v2 import *
import math
#from training_lightGCN_add_fet_tafeng_v35_ser import * # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import normalize
from torch.nn.utils.rnn import pad_sequence
from offered_courses_v2 import *

logging.info("✔︎ GenRec Model Training...")
logger = dh.logger_fn("torch-log", "logs/training-{0}.log".format(time.asctime()))

dilim = '-' * 120
logger.info(dilim)
for attr in sorted(Config().__dict__):
    logger.info('{:>50}|{:<50}'.format(attr.upper(), Config().__dict__[attr]))
logger.info(dilim)

def measure_sequence_of_courses(data, reversed_item_dict):
    #users = data.userID.values
    #print(users)
    #time_baskets = data.baskets.values
    # item_list = []
    # for baskets in data['baskets']:
    #     for basket in baskets:
    #         for item in basket:
    #             if item not in item_list:
    #                 item_list.append(item)
    num_items = len(reversed_item_dict)
    #sequence_dict = {}
    #count_item= {}
    #index1= 0
    seq_matrix = np.zeros((num_items, num_items))
    for baskets in data['baskets']:
        for index1 in range(0, len(baskets)-1):
            for index2 in range(index1+1, len(baskets)):
                list1= baskets[index1]
                list2= baskets[index2]
                for item1 in list1:
                    for item2 in list2:
                        #sequence_dict[item1, item2]= sequence_dict.get((item1, item2),0)+ 1 
                        seq_matrix[reversed_item_dict[item1]][reversed_item_dict[item2]] += 1

    seq_matrix = normalize(seq_matrix, norm='l2') 
    return seq_matrix

def update_idx_of_item_embedding(one_hot_encoded_data, num_users_train, item_embeddings, item_dict_idx_to_cid, item_dict_cid_to_idx, item_dict_one_hot):
    # num_nodes = data.x.size(0)
    
    #num_nodes = final_x.size(0)
    # Initialize new user features as the mean of the features of interacted items
    #item_embeddings = final_x[num_users:]
    interaction_matrix = np.array(one_hot_encoded_data)
    #num_items = num_nodes // 2  # Assuming first half are users and second half are items
    #item_embeddings2 = np.zeros((item_embeddings.shape[0],item_embeddings.shape[1]))
    item_embeddings2 = torch.zeros_like(item_embeddings, dtype=torch.float32)
    for idx1, it_embed in enumerate(item_embeddings):
        cid = item_dict_one_hot[idx1] # idx to cid in LightGCN
        idx2 = item_dict_cid_to_idx[cid] # cid to idx in training data
        item_embeddings2[idx2] = it_embed

    
#     #num_users_new = one_hot_encoded_data.shape[0]
#     new_user_interactions_all = []
#     for user in range(num_users_train):
#         new_user_interactions = []
#         for item in range(num_items):
#             if interaction_matrix[user, item] == 1:
#                 new_user_interactions.append(item)
#         new_user_interactions_all.append(new_user_interactions)
    
#    # new_user_interactions2 = []
#    # for item_idx1 in new_user_interactions:
#         #cid = item_dict_one_hot[item_idx1]
#         #new_user_interactions2.append(item_dict_cid_to_idx[cid]) # index of a course in item embeddings we get from topic modeling
#     #new_user_embedding = item_embeddings[new_user_interactions2].mean(dim=0, keepdim=True)
#     new_user_embeddings_all = []
#     for new_user_interactions2 in new_user_interactions_all:
#         # new_user_embedding = np.mean(item_embeddings2[new_user_interactions2], axis=0)[np.newaxis, :]
#         new_user_embedding = np.mean(item_embeddings2[new_user_interactions2], axis=0).flatten()
#         new_user_embeddings_all.append(new_user_embedding)
#     new_user_embeddings2 = np.array(new_user_embeddings_all)

    # Update node features
    #updated_x = torch.cat([data.x, new_user_embedding], dim=0)
      
    
    # Update edge index to include new edges between the new user and interacted items
    #new_user_index_all = []
    #new_user_index = torch.tensor([[num_nodes] * len(new_user_interactions),  [item_idx + num_users for item_idx in new_user_interactions]])
        #new_user_index_all.append(new_user_index)
    #updated_edge_index = torch.cat([data.edge_index, new_user_index], dim=1)
    
    # Create new Data object with updated features and edges
    #updated_data = Data(x=updated_x, edge_index=updated_edge_index)
    
    return item_embeddings2

def calculate_term_dict_2(term_dict, semester, basket, reversed_item_dict):
    for item in basket:
        if semester not in term_dict:
            count_course = {}
        else:
            count_course = term_dict[semester]
        if item not in count_course:
            count_course[item] = 1
        else:
            count_course[item] = count_course[item]+ 1
        term_dict[semester] = count_course
    return term_dict

def calculate_avg_n_actual_courses(input_data, reversed_item_dict):
    data = input_data
    frequency_of_courses = {}
    for baskets in data["baskets"]:
        for basket in baskets:
            for item in basket:
                if item not in frequency_of_courses:
                    frequency_of_courses[item] = 1
                else:
                    frequency_of_courses[item] += 1
    term_dict_all = {}
    for x in range(len(data)):
        baskets = data['baskets'][x]
        ts = data['timestamps'][x]
        #index1 =0 
        for x1 in range(len(ts)):
            basket = baskets[x1]
            semester = ts[x1]
            term_dict_all = calculate_term_dict_2(term_dict_all, semester, basket, reversed_item_dict)
    count_course_all = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in count_course_all:
                count_course_all[item] = [cnt, 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = count_course_all[item]
                cnt1 += cnt
                n1 += 1
                #count_course_all[item] = list1
                count_course_all[item] = [cnt1, n1]
    count_course_avg = {}
    for course, n in count_course_all.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        count_course_avg[course] = float(cnt2/n2)
    #calculate standard deviation
    course_sd = {}
    for keys, values in term_dict_all.items():
        count_course = values
        for item, cnt in count_course.items():
            if item not in course_sd:
                course_sd[item] = [pow((cnt-count_course_avg[item]),2), 1]
            else:
                # list1 = count_course_all[item]
                # list1[0] = list1[0]+ cnt
                # list1[1] = list1[0]+ 1
                cnt1, n1 = course_sd[item]
                cnt1 = cnt1+ pow((cnt-count_course_avg[item]),2)
                n1 += 1
                #count_course_all[item] = list1
                course_sd[item] = [cnt1, n1]
    course_sd_main = {}
    course_number_terms = {}
    for course, n in course_sd.items():
        #count_course_avg[course] = float(n[0]/n[1])
        cnt2, n2 = n
        if(n2==1): course_sd_main[course] = float(math.sqrt(cnt2/n2))
        else: course_sd_main[course] = float(math.sqrt(cnt2/(n2-1)))
        course_number_terms[course] = n2
    
    return term_dict_all, frequency_of_courses, count_course_avg, course_sd_main, course_number_terms

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
    print(one_hot_df.head)
    
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

def convert_side_info_to_one_hot_encoding(data, reversed_item_dict_one_hot, num_items, min1, bel_avg, avg1, max1):
    # df["category"] 
    # df["level"] 
    #  # ['itemID', 'PRODUCT_SUBCLASS', 'SALES_PRICE']
    cat_f_all = {}
    price_f_all = {}
    for idx1, cid in enumerate(data["itemID"]):
        #index = data[data['userID'] == user].index.values[0]
        if int(cid) in reversed_item_dict_one_hot: 
            idx2 = reversed_item_dict_one_hot[int(cid)] # cid to idx
            cid2 = int (data["itemID"][idx2])
            cat_f = data["PRODUCT_SUBCLASS"][idx2]
            price = data["SALES_PRICE"][idx2]
            if price>=min1 and price <= bel_avg:
                price_f = "low"
            elif price>=bel_avg and price <= avg1:
                price_f = "medium"
            elif price>=avg1 and price <= max1:
                price_f = "high"
            #level_f = data["SALES_PRICE"][idx2]
            row1 = [cid2, cat_f]
            row2 = [cid2, price_f]
            cat_f_all[idx2] = row1
            price_f_all[idx2] = row2

        # cat_f = data["category"][idx1]
        # level_f = data["level"][idx1]
        # row1 = [cid, cat_f]
        # row2 = [cid, str(level_f)]
        # # cat_f_all.append(row1)
        # # level_f_all.append(row2)
        # cat_f_all[idx1] = row1
        # level_f_all[idx1] = row2
    cat_f_all_sorted = dict(sorted(cat_f_all.items(), key=lambda item: item[0], reverse=False))
    level_f_all_sorted = dict(sorted(price_f_all.items(), key=lambda item: item[0], reverse=False))
    cat_f_all = list(cat_f_all_sorted.values())
    level_f_all = list(level_f_all_sorted.values())
    
    cat_f_all_df = pd.DataFrame(cat_f_all, columns=['item', 'cat_f'])
    level_f_all_df = pd.DataFrame(level_f_all, columns=['item', 'level_f'])
    one_hot_encoded_cat, one_hot_df_cat, item_dict_one_hot_cat, cat_dict_one_hot = convert_to_one_hot_encoding_cat(cat_f_all_df) # category = product sublcass
    one_hot_encoded_level, one_hot_df_level, item_dict_one_hot_level, level_dict_one_hot = convert_to_one_hot_encoding_level(level_f_all_df) # level = sales price

    reversed_dict_cat_to_idx = dict(zip(cat_dict_one_hot.values(), cat_dict_one_hot.keys()))  # cat, idx
    reversed_dict_level_to_idx = dict(zip(level_dict_one_hot.values(), level_dict_one_hot.keys()))  # level, idx

    return one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx,  reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level

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
            p_k = 1 if predicted[i] in ground_truth else 0
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
    dcg = dcg_at_k(predicted, ground_truth, k)
    idcg = idcg_at_k(ground_truth, k)
    return dcg / idcg if idcg > 0 else 0.0

def round_based_on_decimal(value):
    decimal_part = value - int(value)
    if decimal_part >= 0.5:
        return math.ceil(value)
    else:
        return math.floor(value)


def train(offered_courses, train_set_without_target, target_set, item_dict, train_data_all, valid_set_without_target, emb_dim, n_attn_lays, attn_drops, attn_lr, n_heads, max_seq_len, max_basket_size, device):
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
    # model = DRModel(Config(), emb_dim, n_attn_lays, attn_drops, attn_lr, device = device)
    # model = model.to(device)
    num_items = len(item_dict)+1
    model = GenRec(Config(), num_items, max_basket_size, emb_dim, n_heads, n_attn_lays, max_seq_len, attn_drops, device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=attn_lr)

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
            #du_p_product = torch.mm(du, item_embedding.t())  # shape: [pad_len, num_item], on device
            du_p_product = torch.mm(du, item_embedding.t()) 
            loss_u = []  # Loss for each user
            
            for t, basket_t in enumerate(bks):
                if basket_t[0] != 0 and t != 0:
                    # Positive indices (basket items)
                    basket_t_main = [item for item in basket_t if item>0] # remove padded item
                    # pos_idx = torch.tensor(basket_t).to(device)  # Move to device
                    pos_idx = torch.tensor(basket_t_main).to(device)  # Move to device

                    # Sample negative products
                    # neg = random.sample(list(neg_samples[uid.item()]), len(basket_t))
                    neg = random.sample(list(neg_samples[uid.item()]), len(basket_t_main))
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
    
    # def compute_bpr_loss_with_decoder(uids, logits, target_seq, neg_samples, device): # uids, baskets, dynamic_user, item_embedding, neg_samples,
    #     """
    #     Compute BPR loss using decoder outputs.
        
    #     Args:
    #         logits: Predicted scores from the decoder (shape: [batch_size, target_seq_len, num_items]).
    #         target_seq: Target items (shape: [batch_size, target_seq_len]).
    #         neg_samples: Dictionary of negative samples for each user.
    #         device: Torch device.
            
    #     Returns:
    #         bpr_loss: Computed BPR loss.
    #     """
    #     bpr_loss = torch.tensor(0.0, device=device)
    #     batch_size, target_len, num_items = logits.shape

    #     for i in range(batch_size):
    #     #for uid, bks, du in zip(uids, baskets, dynamic_user):
    #         pos_indices = target_seq[i]  # Target item indices for user i (shape: [target_len])
    #         pos_scores = logits[i, torch.arange(target_len), pos_indices]  # Shape: [target_len]
    #         uid = uids[i]
    #         # Sample negative items
    #         neg_indices = torch.tensor(
    #             random.sample(neg_samples[uid.item()], target_len), device=device
    #         )  # Shape: [target_len]
    #         # neg = random.sample(list(neg_samples[uid.item()]), len(basket_t))
    #         neg_scores = logits[i, torch.arange(target_len), neg_indices]  # Shape: [target_len]

    #         # Compute BPR loss for user i
    #         loss = torch.mean(-torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    #         bpr_loss += loss

    #     bpr_loss /= batch_size  # Average over the batch
    #     return bpr_loss


    def train_model():
        model.train()  # turn on training mode for dropout
        #dr_hidden = model.init_hidden(Config().batch_size).to(device)
        train_loss = 0
        #train_recall = 0.0
        start_time= time.perf_counter()
       #start_time = time.clock()

        num_batches = ceil(len(train_data) / Config().batch_size)
        # item_seq, basket_seq, target_seq
        for i, x in enumerate(dh.batch_iter(train_data, Config().batch_size, Config().seq_len, num_items, max_basket_size, device, shuffle=True)):
            # uids, baskets, lens, prev_idx = x
            uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = x
            # print("shape of basket_seq: ", len(basket_seq))
            # print("shape of basket_seq 0: ", len(basket_seq[0]))
            # print("shape of basket_seq 00: ", len(basket_seq[0][0]))
            # print("shape of basket_seq 01: ", len(basket_seq[0][1]))
            # print("Max index in item_seq:", item_seq.max().item())
            # print("Min index in item_seq:", item_seq.min().item())
            #print("Embedding num_embeddings:", self.item_embedding.num_embeddings)
                  
            #baskets = [[torch.tensor(basket, device=device) for basket in user] for user in baskets]
            # item_seq = [[torch.tensor(item, device=device) for item in user] for user in item_seq]
            # basket_seq = [[torch.tensor(basket, device=device) for basket in user] for user in basket_seq]
            # target_seq = [[torch.tensor(t_item, device=device) for t_item in user] for user in target_seq]
            # Convert and pad item_seq
            item_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in item_seq]
            item_seq_padded = pad_sequence(item_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_num_items]

            # Convert and pad basket_seq
            basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in basket_seq]
            basket_seq_padded = pad_sequence(basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
            
            all_basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in baskets]
            #all_basket_seq_padded = pad_sequence(all_basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]

            # Convert and pad target_seq
            target_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in target_seq]
            target_seq_padded = pad_sequence(target_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_target_len]
            # position_ids = torch.arange(max_basket_size).unsqueeze(0).expand(max_seq_len, -1)
            position_ids = torch.arange(max_basket_size)
            basket_ids = torch.arange(max_seq_len)
            
            model.zero_grad() 
            # dynamic_user, _ = model(baskets, lens, dr_hidden)
            #dynamic_user = model(item_seq, basket_seq, target_seq)
            dynamic_user = model(basket_seq_padded, position_ids, basket_ids, Config().batch_size, lens, target_seq_padded)
            # loss = bpr_loss(uids, baskets, dynamic_user, model.encode.weight)
            #loss = bpr_loss(uids, baskets, dynamic_user, model.encode)
            # loss = bpr_loss(uids, baskets, dynamic_user, model.encode, neg_samples, device=device)
            print("training done for a batch of inputs")
            loss = bpr_loss(uids, all_basket_seq_tensors, dynamic_user, model.item_embedding.weight, neg_samples, device=device)
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
        #dr_hidden = model.init_hidden(Config().batch_size)  # Initialize hidden state
        #dr_hidden = tuple(h.to(device) for h in dr_hidden) if isinstance(dr_hidden, tuple) else dr_hidden.to(device)
        val_loss = 0
        start_time = time.perf_counter()

        num_batches = ceil(len(validation_data) / Config().batch_size)
        with torch.no_grad():  # Disable gradient computation for validation
            for i, x in enumerate(dh.batch_iter(validation_data, Config().batch_size, Config().seq_len, num_items, max_basket_size, device, shuffle=False)):
                #uids, baskets, lens, prev_idx = x
                uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = x
                 # Convert and pad basket_seq
                basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in basket_seq]
                basket_seq_padded = pad_sequence(basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
                target_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in target_seq]
                target_seq_padded = pad_sequence(target_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_target_len]
                position_ids = torch.arange(max_basket_size).unsqueeze(0).expand(max_seq_len, -1)
                basket_ids = torch.arange(max_seq_len)
                all_basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in baskets]
                #all_basket_seq_padded = pad_sequence(all_basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
                #dynamic_user = model(item_seq, basket_seq, target_seq)
                dynamic_user = model(basket_seq_padded, position_ids, basket_ids, Config().batch_size, lens, target_seq_padded)
                loss = bpr_loss(uids, all_basket_seq_tensors, dynamic_user, model.item_embedding.weight, neg_samples, device=device)
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
    def test_model():
        model.eval()
        item_embedding = model.item_embedding.weight
        item_embedding = item_embedding.to(device)
        #item_embedding = model.encode
        #item_embedding = model.encode.to(device)  # Move item embeddings to device
        #dr_hidden = model.init_hidden(Config().batch_size)
        #dr_hidden = tuple(h.to(device) for h in dr_hidden) if isinstance(dr_hidden, tuple) else dr_hidden.to(device)
        #dr_hidden = model.init_hidden(Config().batch_size)

        hitratio_numer_n = 0
        hitratio_denom = 0
        #ndcg = 0.0
        # recall = 0.0
        # recall_2= 0.0
        # recall_temp = 0.0
        count=0
        count_at_least_one_cor_pred = 0
        count_at_least_one_cor_pred2 = 0
        #recall_test_for_one_cor_pred2 = 0.0
        count_at_least_one_cor_pred3 = 0
        recall_test_main_n= 0.0
        recall_test_main_10= 0.0
        recall_test_main_20= 0.0
        sum_ndcg_at_n= 0.0
        sum_ndcg_at_10= 0.0
        sum_ndcg_at_20= 0.0
        ndcg_at_n = 0.0
        ndcg_at_10 = 0.0
        ndcg_at_20 = 0.0
        num_items = len(item_dict)+1
        #print(target_data)
        for i, x in enumerate(dh.batch_iter(train_set_without_target, Config().batch_size, Config().seq_len, num_items, max_basket_size, device, shuffle=False)):
            # uids, baskets, lens, prev_idx = x
            # dynamic_user, _ = model(baskets, lens, dr_hidden)
            uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = x
            #dynamic_user = model(item_seq, basket_seq, target_seq)
            basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in basket_seq]
            basket_seq_padded = pad_sequence(basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
            target_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in target_seq]
            target_seq_padded = pad_sequence(target_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_target_len]
            position_ids = torch.arange(max_basket_size).unsqueeze(0).expand(max_seq_len, -1)
            basket_ids = torch.arange(max_seq_len)
            #all_basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in baskets]
            #all_basket_seq_padded = pad_sequence(all_basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
            #dynamic_user = model(item_seq, basket_seq, target_seq)
            dynamic_user = model(basket_seq_padded, position_ids, basket_ids, Config().batch_size, lens, target_seq_padded)
            for uid, l, du in zip(uids, lens, dynamic_user):
                scores = []
                #du_latest = du[l - 1].unsqueeze(0)
                du_latest = du[l - 1].unsqueeze(0).to(device)
                user_baskets = train_set_without_target[train_set_without_target['userID'] == uid.item()].baskets.values[0]
                target_semester = target_data[target_data['userID'] == uid.item()].last_semester.values[0]
                positives = target_data[target_data['userID'] == uid.item()].baskets.values[0]  # list dim 1
                positives = set(positives)
                positives = list(positives)
                p_length = len(positives)
                item_list1 = [i for i in range(num_items)]
                #item_list_ten = torch.LongTensor(item_list1).to(device)
                item_list_ten = torch.tensor(item_list1).to(device)
                #scores = list(torch.mm(du_latest, item_embedding[item_list_ten].t()).data.numpy()[0])
                scores = torch.mm(du_latest, item_embedding[item_list_ten].t()).squeeze(0)
                # for s in scores_all:
                #     scores.append(s)

                # item_list1= []
                # # calculating <u,p> score for all test items <u,p> pair
                # positives = target_data[target_data['userID'] == uid].baskets.values[0]  # list dim 1
                # #target_semester = target_data[target_data['userID'] == uid].last_semester.values[0]

                # for x1 in positives:
                #     item_list1.append(x1)
                # #print(positives)

                # #p_length = len(positives)
                # #t_length = len(positives)
                # positives2 = torch.LongTensor(positives)
                # #print(positives)
                # # Deal with positives samples
                # scores_pos = list(torch.mm(du_latest, item_embedding[positives2].t()).data.numpy()[0])
                # for s in scores_pos:
                #     scores.append(s)
                
                # positives_prior = [item for bskt in user_baskets for item in bskt]
                # for x3 in positives_prior:
                #     item_list1.append(x3)
                # #t_length = len(positives)
                # positives2_prior = torch.LongTensor(positives_prior)
                # scores_pos_prior = list(torch.mm(du_latest, item_embedding[positives2_prior].t()).data.numpy()[0])
                # for s in scores_pos_prior:
                #     scores.append(s)
                

                # # Deal with negative samples
                # negtives = list(neg_samples[uid]) #taking all the available items 
                # #negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
                # for x2 in negtives:
                #     item_list1.append(x2)
                # negtives2 = torch.LongTensor(negtives)
                # scores_neg = list(torch.mm(du_latest, item_embedding[negtives2].t()).data.numpy()[0])
                # for s in scores_neg:
                #     scores.append(s)
                #print(item_list1)
                #print(scores)
                # Calculate hit-ratio
               # index_k = []
                #top_k1= Config().top_k
                top_k1 = len(positives)
                top_k_count= 0
                top_k_count2= 0
                top_k_count3= 0
                #list_key= []
                list1= []
                list2= []
                list3 =[]
                top_k10= 10
                top_k20 = 20
                # repeat_ratio_n = round_based_on_decimal(repeat_ratio * top_k1)
                # repeat_ratio_10 = round_based_on_decimal(repeat_ratio * top_k10)
                # repeat_ratio_20 = round_based_on_decimal(repeat_ratio * top_k20)
                # new_items_ratio_n = top_k1 - repeat_ratio_n
                # new_items_ratio_10 = top_k10 - repeat_ratio_10
                # new_items_ratio_20 = top_k20 - repeat_ratio_20

                cnt_repeated_rec_n = 0
                cnt_repeated_rec_10 = 0
                cnt_repeated_rec_20 = 0
                cnt_new_item_rec_n = 0
                cnt_new_item_rec_10 = 0
                cnt_new_item_rec_20 = 0
                #print(offered_courses[l+1])
                #k=0
                #pred_items= []
                count1= 0
                #print(offered_courses)
                #max_iter1 = max(top_k1, top_k20)

                _, sorted_indices = torch.sort(scores, descending=True)
                for index in sorted_indices:
                #index = scores.index(max(scores))
                # item_id1 = item_list1[index]
                    item_id1 = item_list1[index.item()]
                    if item_id1!= 0 and not utils_CDREAM.filtering(item_id1, user_baskets, offered_courses[target_semester], item_dict):
                    #if not utils_tafeng_CDREAM.filtering(item_id1, user_baskets, item_dict):  # not repeated item = new item
                        if item_id1 not in list1:
                            if top_k_count<top_k1:
                                list1.append(item_id1)
                                top_k_count += 1
                                #cnt_new_item_rec_n += 1
                        if item_id1 not in list2:
                            if top_k_count2<top_k10:
                                list2.append(item_id1)
                                top_k_count2 += 1
                                #cnt_new_item_rec_10 += 1
                        if item_id1 not in list3:
                            if top_k_count3<top_k20:
                                list3.append(item_id1)
                                top_k_count3 += 1
                                #cnt_new_item_rec_20 += 1
                                    #list2.append(index_j)
                                #top_k_count+= 1
                        if(top_k1>= top_k20 and top_k_count==top_k1 and top_k_count3== top_k20): break
                        elif (top_k1< top_k20 and top_k_count3==top_k20): break
                # print("count of iter over train data to rec top_k1 and top_k10 and topk_20 items to a user: ", count1)
                
                


                #print(index_k)
                #print(pred_items)
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer_n += len((set(positives) & set(list1)))
                hitratio_denom += p_length
                #print(index_k)

                # Calculate NDCG
                # u_dcg = 0
                # u_idcg = 0
                # for k1 in range(Config().top_k):
                #     if index_k[k1] < p_length:  
                #         u_dcg += 1 / math.log(k1 + 1 + 1, 2)
                #     u_idcg += 1 / math.log(k1 + 1 + 1, 2)
                # ndcg += u_dcg / u_idcg
                #calculate recall
                #recall_2+= recall_cal(positives, index_k)
                recall_temp_n, count_at_least_one_cor_pred= recall_cal(positives, list1, count_at_least_one_cor_pred)
                recall_temp_10, count_at_least_one_cor_pred2 = recall_cal(positives, list2, count_at_least_one_cor_pred2)
                recall_temp_20, count_at_least_one_cor_pred3= recall_cal(positives, list3, count_at_least_one_cor_pred3)
                recall_test_main_n += recall_temp_n
                recall_test_main_10 += recall_temp_10
                recall_test_main_20 += recall_temp_20
                ndcg_at_n = ndcg_at_k(list1, positives, p_length)
                ndcg_at_10 = ndcg_at_k(list2, positives, k=10)
                ndcg_at_20 = ndcg_at_k(list3, positives, k =20)
                sum_ndcg_at_n += ndcg_at_n
                sum_ndcg_at_10 += ndcg_at_10
                sum_ndcg_at_20 += ndcg_at_20
                #recall_2+= recall_temp
                count+= 1
                #print("count of iter over train users: ", count)
                
        hit_ratio_n = hitratio_numer_n / hitratio_denom
        #ndcg = ndcg / len(train_data)
        #recall = recall_2/ count
        recall_test_n = recall_test_main_n/ count
        recall_test_10 = recall_test_main_10/ count
        recall_test_20 = recall_test_main_20/ count
        avg_ndcg_at_n = sum_ndcg_at_n/ count
        avg_ndcg_at_10 = sum_ndcg_at_10/ count
        avg_ndcg_at_20 = sum_ndcg_at_20/ count
        logger.info('[Test]| Epochs {:3d} | Hit ratio {:02.4f} | recall {:05.4f} |'
                    .format(epoch, hit_ratio_n, recall_test_n))
        print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
        percentage_of_at_least_one_cor_pred_n = count_at_least_one_cor_pred/ len(target_data)
        print("percentage_of_at_least_one_cor_pred_n: ", percentage_of_at_least_one_cor_pred_n)
        percentage_of_at_least_one_cor_pred_10 = (count_at_least_one_cor_pred2/ count)* 100 # PHR@10
        print("percentage_of_at_least_one_cor_pred_10: ", percentage_of_at_least_one_cor_pred_10)
        percentage_of_at_least_one_cor_pred_20 = (count_at_least_one_cor_pred3/ count)* 100 # PHR@20
        print("percentage_of_at_least_one_cor_pred_20: ", percentage_of_at_least_one_cor_pred_20)

        return hit_ratio_n, recall_test_n, recall_test_10, recall_test_20, percentage_of_at_least_one_cor_pred_n, percentage_of_at_least_one_cor_pred_10, percentage_of_at_least_one_cor_pred_20, avg_ndcg_at_n, avg_ndcg_at_10, avg_ndcg_at_20

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_cr_genrec_v1", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info('Save into {0}'.format(out_dir))
    checkpoint_dir = out_dir + '/model-{epoch:02d}-{hitratio:.4f}-{recall:.4f}.model'

    best_hit_ratio = None
    lowest_val_loss = None
    hit_ratio_n = 0.0
    recall_test_n = 0.0

    try:
        # Training
        for epoch in range(Config().epochs):
            train_model()
            logger.info('-' * 89)

            val_loss = validate_model()
            logger.info('-' * 89)
            # Checkpoint
            if not lowest_val_loss or val_loss < lowest_val_loss:
                with open(checkpoint_dir.format(epoch=epoch, hitratio=hit_ratio_n, recall=recall_test_n), 'wb') as f:
                    torch.save(model, f)
                lowest_val_loss = val_loss

        hit_ratio_n, recall_test_n, recall_test_10, recall_test_20, percentage_of_at_least_one_cor_pred_n, percentage_of_at_least_one_cor_pred_10, percentage_of_at_least_one_cor_pred_20, avg_ndcg_at_n, avg_ndcg_at_10, avg_ndcg_at_20 = test_model()
        logger.info('-' * 89)

        # # Checkpoint
        # if not best_hit_ratio or hit_ratio_n > best_hit_ratio:
        #     with open(checkpoint_dir.format(epoch=epoch, hitratio=hit_ratio_n, recall=recall_test_n), 'wb') as f:
        #         torch.save(model, f)
        #     best_hit_ratio = hit_ratio_n
    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')
    print("model directory: ", timestamp)
    #print("config for train: 64, 2, 0.6")

    return timestamp, hit_ratio_n, recall_test_n, recall_test_10, recall_test_20, percentage_of_at_least_one_cor_pred_n, percentage_of_at_least_one_cor_pred_10, percentage_of_at_least_one_cor_pred_20, avg_ndcg_at_n, avg_ndcg_at_10, avg_ndcg_at_20

# def recall_cal(positives, pred_items):
#         p_length= len(positives)
#         #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
#         correct_preds= len((set(positives) & set(pred_items))) #total number of matches
#         #print(correct_preds)
#         actual_bsize= p_length
#         return float(correct_preds/actual_bsize)
#         #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))
def recall_cal(positives, pred_items, count_at_least_one_cor_pred):
        p_length= len(positives)
        #correct_preds= len((set(np.arange(0, p_length)) & set(index_k2))) #total number of matches 
        correct_preds= len((set(positives) & set(pred_items))) #total number of matches
        #print(correct_preds)
        actual_bsize= p_length
        if(correct_preds>=1): count_at_least_one_cor_pred += 1
        #return tf.reduce_mean(tf.cast(correct_preds, dtype=tf.float32) / tf.cast(actual_bsize, dtype=tf.float32))
        return float(correct_preds/actual_bsize), count_at_least_one_cor_pred
# validation recall considering prereq connections
def valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, valid_data, valid_target, max_basket_size, device):
    f = open(output_path, "w") #generating text file with recommendation using filtering function
    # Load data
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    #test_data = dh.load_data(Config().TRAININGSET_DIR)
    # valid_data = dh.load_data('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/valid_sample_without_target.json')

    logger.info("✔︎ Test data processing...")
    #test_target = dh.load_data(Config().TESTSET_DIR)
    #valid_target = dh.load_data('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/validation_target_set.json')

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    #dr_model = torch.load(MODEL_DIR)
    #device = torch.device("cpu")
    # checkpoint = torch.load(MODEL_DIR, map_location=device)
    # dr_model = DRModel(Config(), n_rnn_lays, rnn_drops, rnn_lr, device)
    # dr_model.load_state_dict(checkpoint['model_state_dict'])
    dr_model = torch.load(MODEL_DIR)
    dr_model = dr_model.to(device)  # Move to the appropriate devic

    dr_model.eval()

    item_embedding = dr_model.item_embedding.weight
    #item_embedding = dr_model.encode
    #hidden = dr_model.init_hidden(Config().batch_size)
    #dr_hidden = model.init_hidden(Config().batch_size)
    #hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)

    #hitratio_numer = 0
   # hitratio_denom = 0
    #ndcg = 0.0
    #recall = 0.0
   # recall_2= 0.0
    #recall_3= 0.0
    #count=0
    hitratio_numer_n = 0
    hitratio_denom = 0
    #ndcg = 0.0
    # recall = 0.0
    # recall_2= 0.0
    # recall_temp = 0.0
    count=0
    count_at_least_one_cor_pred = 0
    count_at_least_one_cor_pred2 = 0
    #recall_test_for_one_cor_pred2 = 0.0
    count_at_least_one_cor_pred3 = 0
    recall_test_main_n= 0.0
    recall_test_main_10= 0.0
    recall_test_main_20= 0.0
    sum_ndcg_at_n= 0.0
    sum_ndcg_at_10= 0.0
    sum_ndcg_at_20= 0.0
    ndcg_at_n = 0.0
    ndcg_at_10 = 0.0
    ndcg_at_20 = 0.0
    recall_test_main_n_wcrr= 0.0
    recall_test_main_10_wcrr= 0.0
    recall_test_main_20_wcrr= 0.0
    sum_ndcg_at_n_wcrr= 0.0
    sum_ndcg_at_10_wcrr= 0.0
    sum_ndcg_at_20_wcrr= 0.0
    PHR_at_n_wcrr = 0.0
    PHR_at_10_wcrr = 0.0
    PHR_at_20_wcrr = 0.0
    ndcg_at_n_wcrr = 0.0
    ndcg_at_10_wcrr = 0.0
    ndcg_at_20_wcrr = 0.0
    # recall_bsize = {}
    # missed_bsize = {}
    # retake_bsize = {}
    # non_CIS_bsize = {}
    # CIS_missed_bsize = {}
    #test_recall = 0.0
    #last_batch_actual_size = len(valid_data) % Config().batch_size
    num_items = len(item_dict)+1
    for i, x in enumerate(dh.batch_iter(valid_data, Config().batch_size, Config().seq_len, num_items, max_basket_size, device, shuffle=False)):
        # uids, baskets, lens, prev_idx = x
        # dynamic_user, _ = dr_model(baskets, lens, hidden)
        uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = x
        #dynamic_user = model(item_seq, basket_seq, target_seq)
        basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in basket_seq]
        basket_seq_padded = pad_sequence(basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
        target_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in target_seq]
        target_seq_padded = pad_sequence(target_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_target_len]
        position_ids = torch.arange(max_basket_size).unsqueeze(0).expand(max_seq_len, -1)
        basket_ids = torch.arange(max_seq_len)
        #all_basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in baskets]
        #all_basket_seq_padded = pad_sequence(all_basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
        #dynamic_user = model(item_seq, basket_seq, target_seq)
        # dynamic_user = dr_model(all_basket_seq_padded, position_ids, basket_ids, Config().batch_size, lens)
        dynamic_user = dr_model(basket_seq_padded, position_ids, basket_ids, Config().batch_size, lens, target_seq_padded)
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            scores = []
            du_latest = du[l - 1].unsqueeze(0).to(device)
            user_baskets = valid_data[valid_data['userID'] == uid.item()].baskets.values[0]
            #prior_bsize = len(user_baskets)
            #print("user_baskets: ", user_baskets)
            positives = valid_target[valid_target['userID'] == uid.item()].baskets.values[0]  # list dim 1
            target_semester = valid_data[valid_data['userID'] == uid.item()].last_semester.values[0]
            # target_semester = valid_target['last_semester'][i]
            positives = set(positives)
            positives = list(positives)
            p_length = len(positives)
            item_list1 = [i for i in range(num_items)]
            item_list_ten = torch.tensor(item_list1).to(device)
            scores = torch.mm(du_latest, item_embedding[item_list_ten].t()).squeeze(0)
            #scores = list(torch.mm(du_latest, item_embedding[item_list_ten].t()).data.numpy()[0])
            #item_list1= []
            # calculating <u,p> score for all test items <u,p> pair
            # positives = valid_target[valid_target['userID'] == uid].baskets.values[0]  # list dim 1
            # #target_semester = valid_target[valid_target['userID'] == uid].last_semester.values[0]
            # #print("uid: ", uid, " ",positives)
            # for x1 in positives:
            #     item_list1.append(x1)
            # #print(positives)

            # p_length = len(positives)
            # positives2 = torch.LongTensor(positives)
            # #print(positives)
            # # Deal with positives samples
            # scores_pos = list(torch.mm(du_latest, item_embedding[positives2].t()).data.numpy()[0])
            # for s in scores_pos:
            #     scores.append(s)
            
            # positives_prior = [item for bskt in user_baskets for item in bskt]
            # for x3 in positives_prior:
            #     item_list1.append(x3)
            # #t_length = len(positives)
            # positives2_prior = torch.LongTensor(positives_prior)
            # scores_pos_prior = list(torch.mm(du_latest, item_embedding[positives2_prior].t()).data.numpy()[0])
            # for s in scores_pos_prior:
            #     scores.append(s)

            # # Deal with negative samples
            # #negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
            # negtives = list(neg_samples[uid])
            # for x2 in negtives:
            #     item_list1.append(x2)
            # negtives2 = torch.LongTensor(negtives)
            # scores_neg = list(torch.mm(du_latest, item_embedding[negtives2].t()).data.numpy()[0])
            # for s in scores_neg:
            #     scores.append(s)
            #print(item_list1)
            #print(scores)
            # Calculate hit-ratio
            #index_k = []
            #top_k1= Config().top_k
            top_k1 = len(positives)
            top_k_count= 0
            top_k_count2= 0
            top_k_count3= 0
            #list_key= []
            list1= []
            list2= []
            list3 =[]
            list4= []
            list5= []
            list6 =[]
            top_k10= 10
            top_k20 = 20
            # repeat_ratio_n = round_based_on_decimal(repeat_ratio * top_k1)
            # repeat_ratio_10 = round_based_on_decimal(repeat_ratio * top_k10)
            # repeat_ratio_20 = round_based_on_decimal(repeat_ratio * top_k20)
            # new_items_ratio_n = top_k1 - repeat_ratio_n
            # new_items_ratio_10 = top_k10 - repeat_ratio_10
            # new_items_ratio_20 = top_k20 - repeat_ratio_20

            cnt_repeated_rec_n = 0
            cnt_repeated_rec_10 = 0
            cnt_repeated_rec_20 = 0
            cnt_new_item_rec_n = 0
            cnt_new_item_rec_10 = 0
            cnt_new_item_rec_20 = 0
            #print(offered_courses[l+1])
            if t_idx==1: # we are not considering randomly selected instances for last batch
                #k=0
                #pred_items= []
                count1= 0
                _, sorted_indices = torch.sort(scores, descending=True)
                list4= sorted_indices[:top_k1]
                list5= sorted_indices[:top_k10]
                list6 = sorted_indices[:top_k20] 
                list4 = list4.tolist()
                list5 = list5.tolist()
                list6 = list6.tolist()
            #for index in sorted_indices:
            #index = scores.index(max(scores))
            # item_id1 = item_list1[index]
            #count1 = 0
            for index in sorted_indices:
                #index = scores.index(max(scores))
                # item_id1 = item_list1[index]
                item_id1 = item_list1[index.item()]
                if item_id1!= 0 and not utils_CDREAM.filtering(item_id1, user_baskets, offered_courses[target_semester], item_dict):
                #if not utils_tafeng_CDREAM.filtering(item_id1, user_baskets, item_dict):  # not repeated item = new item
                    if item_id1 not in list1:
                        if top_k_count<top_k1:
                            list1.append(item_id1)
                            top_k_count += 1
                            #cnt_new_item_rec_n += 1
                    if item_id1 not in list2:
                        if top_k_count2<top_k10:
                            list2.append(item_id1)
                            top_k_count2 += 1
                            #cnt_new_item_rec_10 += 1
                    if item_id1 not in list3:
                        if top_k_count3<top_k20:
                            list3.append(item_id1)
                            top_k_count3 += 1
                            #cnt_new_item_rec_20 += 1
                                #list2.append(index_j)
                            #top_k_count+= 1
                    if(top_k1>= top_k20 and top_k_count==top_k1 and top_k_count3== top_k20): break
                    elif (top_k1< top_k20 and top_k_count3==top_k20): break
                    #scores[index] = -999999
                    # count1+= 1
                    # if(count1==len(scores)): break
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
                for item3 in list1:
                    f.write(str(reversed_item_dict[item3])+ " ")
                    rec_basket2.append(reversed_item_dict[item3])
                f.write("\n") 
                    
                prior_courses = []
                for basket3 in user_baskets:
                    for item4 in basket3:
                        if reversed_item_dict[item4] not in prior_courses:
                            prior_courses.append(reversed_item_dict[item4])
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer_n += len((set(positives) & set(list1)))
                hitratio_denom += p_length
                #print(index_k)

                # Calculate NDCG
                # u_dcg = 0
                # u_idcg = 0
                # for k1 in range(Config().top_k):
                #     if index_k[k1] < p_length:  
                #         u_dcg += 1 / math.log(k1 + 1 + 1, 2)
                #     u_idcg += 1 / math.log(k1 + 1 + 1, 2)
                # ndcg += u_dcg / u_idcg
                #calculate recall
                #recall_2+= recall_cal(positives, index_k)
                recall_temp_n, count_at_least_one_cor_pred= recall_cal(positives, list1, count_at_least_one_cor_pred)
                recall_temp_10, count_at_least_one_cor_pred2 = recall_cal(positives, list2, count_at_least_one_cor_pred2)
                recall_temp_20, count_at_least_one_cor_pred3= recall_cal(positives, list3, count_at_least_one_cor_pred3)
                recall_test_main_n += recall_temp_n
                recall_test_main_10 += recall_temp_10
                recall_test_main_20 += recall_temp_20
                ndcg_at_n = ndcg_at_k(list1, positives, p_length)
                ndcg_at_10 = ndcg_at_k(list2, positives, k=10)
                ndcg_at_20 = ndcg_at_k(list3, positives, k =20)
                sum_ndcg_at_n += ndcg_at_n
                sum_ndcg_at_10 += ndcg_at_10
                sum_ndcg_at_20 += ndcg_at_20
                recall_temp_n_wcrr, PHR_at_n_wcrr = recall_cal(positives, list4, PHR_at_n_wcrr)
                recall_temp_10_wcrr, PHR_at_10_wcrr = recall_cal(positives, list5, PHR_at_10_wcrr)
                recall_temp_20_wcrr, PHR_at_20_wcrr = recall_cal(positives, list6, PHR_at_20_wcrr)
                recall_test_main_n_wcrr += recall_temp_n_wcrr
                recall_test_main_10_wcrr += recall_temp_10_wcrr
                recall_test_main_20_wcrr += recall_temp_20_wcrr
                ndcg_at_n_wcrr = ndcg_at_k(list4, positives, p_length)
                ndcg_at_10_wcrr = ndcg_at_k(list5, positives, k=10)
                ndcg_at_20_wcrr = ndcg_at_k(list6, positives, k =20)
                sum_ndcg_at_n_wcrr += ndcg_at_n_wcrr
                sum_ndcg_at_10_wcrr += ndcg_at_10_wcrr
                sum_ndcg_at_20_wcrr += ndcg_at_20_wcrr
                #recall_2+= recall_temp
                count+= 1
                # if prior_bsize not in recall_bsize:
                #     recall_bsize[prior_bsize]= [recall_temp]
                # else:
                #     recall_bsize[prior_bsize] += [recall_temp]
                # number of non-CIS and retake courses out of missed courses    
                # n_missed = 0
                # n_retake = 0
                # n_non_CIS =0
                # n_CIS = 0
                # n_non_CIS_all = 0
                # unique_courses = []
                # freq =0
                # for course2 in target_basket2:
                #     if course2 not in rec_basket2:
                #         n_missed += 1
                #         if course_CIS_dept_filtering(course2)==0:
                #             n_non_CIS +=1
                #         else:
                #             n_CIS +=1
                #         if course2 in prior_courses:
                #             n_retake += 1
                #     if course_CIS_dept_filtering(course2)==0:
                #         n_non_CIS_all += 1
                #         if course2 not in unique_courses:
                #             unique_courses.append(course2)
                #         # freq += count_course_avg_train[course2]
                #         freq += count_course_avg_train[course2]
                # if prior_bsize not in non_CIS_bsize:
                #     non_CIS_bsize[prior_bsize]= [n_non_CIS_all, unique_courses, 1, freq]
                # else:
                #     n3, uq, cnt3, fq = non_CIS_bsize[prior_bsize]
                #     for c4 in unique_courses:
                #         if c4 not in uq:
                #             uq.append(c4)
                #     n3 += n_non_CIS_all
                #     cnt3+= 1
                #     fq += freq
                #     non_CIS_bsize[prior_bsize] = [n3, uq, cnt3, fq]
                # if n_missed>0:
                #     if prior_bsize not in missed_bsize:
                #         missed_bsize[prior_bsize]= [n_non_CIS, n_missed]
                #     else:
                #         x3, y3 = missed_bsize[prior_bsize]
                #         x3+= n_non_CIS
                #         y3 += n_missed
                #         missed_bsize[prior_bsize] = [x3, y3]
                    
                #     if prior_bsize not in retake_bsize:
                #         retake_bsize[prior_bsize]= [n_retake, n_missed]
                #     else:
                #         x4, y4 = retake_bsize[prior_bsize]
                #         x4+= n_retake
                #         y4 += n_missed
                #         retake_bsize[prior_bsize] = [x4, y4]
                #     if prior_bsize not in CIS_missed_bsize:
                #         CIS_missed_bsize[prior_bsize]= [n_CIS, n_missed]
                #     else:
                #         x5, y5 = CIS_missed_bsize[prior_bsize]
                #         x5+= n_CIS
                #         y5 += n_missed
                #         CIS_missed_bsize[prior_bsize] = [x5, y5]
                # if n_missed>0:
                #     v3= n_non_CIS/n_missed
                #     if prior_bsize not in missed_bsize:
                #         missed_bsize[prior_bsize]= [v3]
                #     else:
                #         # x3, y3 = missed_bsize[prior_bsize]
                #         # x3+= n_non_CIS
                #         # y3 += n_missed
                #         missed_bsize[prior_bsize] += [v3]
                #     v4= n_retake/n_missed
                #     if prior_bsize not in retake_bsize:
                #         retake_bsize[prior_bsize]= [v4]
                #     else:
                #         # x4, y4 = retake_bsize[prior_bsize]
                #         # x4+= n_retake
                #         # y4 += n_missed
                #         retake_bsize[prior_bsize] += [v4]

    hit_ratio_n = hitratio_numer_n / hitratio_denom
    #ndcg = ndcg / len(train_data)
    #recall = recall_2/ count
    recall_test_n = recall_test_main_n/ count
    recall_test_10 = recall_test_main_10/ count
    recall_test_20 = recall_test_main_20/ count
    avg_ndcg_at_n = sum_ndcg_at_n/ count
    avg_ndcg_at_10 = sum_ndcg_at_10/ count
    avg_ndcg_at_20 = sum_ndcg_at_20/ count

    recall_test_n_wcrr = recall_test_main_n_wcrr/ count
    recall_test_10_wcrr = recall_test_main_10_wcrr/ count
    recall_test_20_wcrr = recall_test_main_20_wcrr/ count
    avg_ndcg_at_n_wcrr = sum_ndcg_at_n_wcrr/ count
    avg_ndcg_at_10_wcrr = sum_ndcg_at_10_wcrr/ count
    avg_ndcg_at_20_wcrr = sum_ndcg_at_20_wcrr/ count
    avg_PHR_at_n_wcrr = (PHR_at_n_wcrr/ count) * 100
    avg_PHR_at_10_wcrr = (PHR_at_10_wcrr/ count) * 100
    avg_PHR_at_20_wcrr = (PHR_at_20_wcrr/ count) * 100
    print(str('Hit ratio[@n]: {0}'.format(hit_ratio_n)))
    f.write(str('Hit ratio[@n]: {0}'.format(hit_ratio_n)))
    f.write("\n")
    #print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    print('Recall[@n]: {0}'.format(recall_test_n))
    f.write(str('Recall[@n]: {0}'.format(recall_test_n)))
    f.write("\n")
    percentage_of_at_least_one_cor_pred_n = count_at_least_one_cor_pred/ len(valid_target)
    print("percentage_of_at_least_one_cor_pred_n: ", percentage_of_at_least_one_cor_pred_n)
    percentage_of_at_least_one_cor_pred_10 = (count_at_least_one_cor_pred2/ count)* 100 # PHR@10
    print("percentage_of_at_least_one_cor_pred_10: ", percentage_of_at_least_one_cor_pred_10)
    percentage_of_at_least_one_cor_pred_20 = (count_at_least_one_cor_pred3/ count)* 100 # PHR@20
    print("percentage_of_at_least_one_cor_pred_20: ", percentage_of_at_least_one_cor_pred_20)

     # recall scores for different basket sizes
    # recall_bsize = dict(sorted(recall_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in recall_bsize.items():
    #     bsize = k
    #     sum = 0
    #     for r in v:
    #         sum += r
    #     recall = sum/len(v)
    #     print("prior basket size: ", bsize)
    #     print("number of instances: ", len(v))
    #     print("recall score for validation data: ", recall)
    #  # number of non_CIS courses out of missed courses for different number of prior semesters
    # missed_bsize = dict(sorted(missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in missed_bsize.items():
    #     bsize = k
    #     tot_non_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     per_of_non_CIS = (tot_non_CIS/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of non CIS courses out of missed courses for validation data: ", per_of_non_CIS)
    # # number of retake courses out of missed courses for different number of prior semesters
    # retake_bsize = dict(sorted(retake_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in retake_bsize.items():
    #     bsize = k
    #     tot_retake, tot_missed = v
    #     per_of_retaken_courses = (tot_retake/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print("percentage of retaken courses out of missed courses for validation data: ", per_of_retaken_courses)

    # non_CIS_bsize = dict(sorted(non_CIS_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in non_CIS_bsize.items():
    #     bsize = k
    #     # sum5 = 0
    #     # for r in v:
    #     #     sum5 += r
    #     sum5, un_c, ct, freq1 = v
    #     avg_non_CIS = sum5/ct
    #     avg_pop = freq1/sum5
    #     print("prior basket size: ", bsize)
    #     print("number of instances: ", ct)
    #     print("total non_CIS_courses: ", sum5)
    #     print("average non_CIS courses for validation data: ",avg_non_CIS)
    #     print("total unique non_CIS courses for validation data: ",len(un_c))
    #     print("average popularity of non_CIS courses for validation data: ", avg_pop)
    #     if bsize ==17: print(un_c)
    
    #  # number of CIS courses missed out of missed courses for different number of prior semesters
    # CIS_missed_bsize = dict(sorted(CIS_missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in CIS_missed_bsize.items():
    #     bsize = k
    #     tot_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     per_of_CIS_missed = (tot_CIS/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of CIS courses out of missed courses: ", per_of_CIS_missed)
    # number of non_CIS courses out of missed courses for different number of prior semesters
    # missed_bsize = dict(sorted(missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in missed_bsize.items():
    #     bsize = k
    #     #tot_non_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     sum3 = 0
    #     for r in v:
    #         sum3 += r
    #     per_of_non_CIS = (sum3/ len(v)) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of non CIS courses out of missed courses for test data: ", per_of_non_CIS)
    # # number of retake courses out of missed courses for different number of prior semesters
    # retake_bsize = dict(sorted(retake_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in retake_bsize.items():
    #     bsize = k
    #     # tot_retake, tot_missed = v
    #     sum4 = 0
    #     for r in v:
    #         sum4 += r
    #     per_of_retaken_courses = (sum4/ len(v)) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print("percentage of retaken courses out of missed courses for test data: ", per_of_retaken_courses)

    f.close()
    return hit_ratio_n, recall_test_n, recall_test_10, recall_test_20, percentage_of_at_least_one_cor_pred_n, percentage_of_at_least_one_cor_pred_10, percentage_of_at_least_one_cor_pred_20, avg_ndcg_at_n, avg_ndcg_at_10, avg_ndcg_at_20, recall_test_n_wcrr, recall_test_10_wcrr, recall_test_20_wcrr, avg_ndcg_at_n_wcrr, avg_ndcg_at_10_wcrr, avg_ndcg_at_20_wcrr, avg_PHR_at_n_wcrr, avg_PHR_at_10_wcrr, avg_PHR_at_20_wcrr

def recall_cal_test(positives, pred_items, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred):
        p_length= len(positives)
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
# testing with CDREAM_LGCN model without considering prereq - original result
def test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_data, test_target, max_basket_size, device):
    f = open(output_path, "w") #generating text file with recommendation using filtering function
    # Load data
    #logger.info("✔︎ Loading data...")

    #logger.info("✔︎ Training data processing...")
    #test_data = dh.load_data(Config().TRAININGSET_DIR)
    # test_data = dh.load_data('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_sample_without_target.json')
    #test_data = dh.load_data('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_sample_without_target.json')

    #logger.info("✔︎ Test data processing...")
    #test_target = dh.load_data(Config().TESTSET_DIR)
    #test_target = dh.load_data('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_target_set.json')
    #test_target_new = dh.load_data('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_target_set_with_more_target_baskets.json')

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    #dr_model = torch.load(MODEL_DIR)

    #dr_model.eval()

    # checkpoint = torch.load(MODEL_DIR, map_location=device)
    # dr_model = DRModel(Config(), n_rnn_lays, rnn_drops, rnn_lr, device)
    # dr_model.load_state_dict(checkpoint['model_state_dict'])
    dr_model = torch.load(MODEL_DIR)
    dr_model = dr_model.to(device)  # Move to the appropriate devic

    dr_model.eval()

    item_embedding = dr_model.item_embedding.weight
    #item_embedding = dr_model.encode
    #hidden = dr_model.init_hidden(Config().batch_size)
    #dr_hidden = model.init_hidden(Config().batch_size)
    #hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)

    # item_embedding = dr_model.encode.weight
    # item_embedding = dr_model.encode
    # hidden = dr_model.init_hidden(Config().batch_size)

    # hitratio_numer = 0
    # hitratio_denom = 0
    # #ndcg = 0.0
    # recall = 0.0
    # recall_2= 0.0
    # #recall_3= 0.0
    # count=0
    hitratio_numer_n = 0
    hitratio_denom = 0
    #ndcg = 0.0
    # recall = 0.0
    # recall_2= 0.0
    # recall_temp = 0.0
    count=0
    count_at_least_one_cor_pred = 0
    count_at_least_one_cor_pred2 = 0
    #recall_test_for_one_cor_pred2 = 0.0
    count_at_least_one_cor_pred3 = 0
    recall_test_main_n= 0.0
    recall_test_main_10= 0.0
    recall_test_main_20= 0.0
    sum_ndcg_at_n= 0.0
    sum_ndcg_at_10= 0.0
    sum_ndcg_at_20= 0.0
    ndcg_at_n = 0.0
    ndcg_at_10 = 0.0
    ndcg_at_20 = 0.0
    recall_test_main_n_wcrr= 0.0
    recall_test_main_10_wcrr= 0.0
    recall_test_main_20_wcrr= 0.0
    sum_ndcg_at_n_wcrr= 0.0
    sum_ndcg_at_10_wcrr= 0.0
    sum_ndcg_at_20_wcrr= 0.0
    PHR_at_n_wcrr = 0.0
    PHR_at_10_wcrr = 0.0
    PHR_at_20_wcrr = 0.0
    ndcg_at_n_wcrr = 0.0
    ndcg_at_10_wcrr = 0.0
    ndcg_at_20_wcrr = 0.0
    #count_at_least_one_cor_pred = 0
    total_correct_preds = 0
    recall_test_for_one_cor_pred = 0.0
    count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred  = 0, 0, 0, 0, 0
    count_at_least_two_cor_pred2, count_at_least_three_cor_pred2, count_at_least_four_cor_pred2, count_at_least_five_cor_pred2, count_all_cor_pred2  = 0, 0, 0, 0, 0
    count_at_least_two_cor_pred3, count_at_least_three_cor_pred3, count_at_least_four_cor_pred3, count_at_least_five_cor_pred3, count_all_cor_pred3  = 0, 0, 0, 0, 0
    count_actual_bsize_at_least_2, count_actual_bsize_at_least_3, count_actual_bsize_at_least_4, count_actual_bsize_at_least_5, count_actual_bsize_at_least_6 = 0, 0, 0, 0, 0
    #recall_temp =0.0
    # target_basket_size = {}
    # target_basket_size[1] = 0
    # target_basket_size[2] = 0
    # target_basket_size[3] = 0
    # target_basket_size[4] = 0
    # target_basket_size[5] = 0
    # target_basket_size[6] = 0
    count_cor_pred = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred[x5,y5] = 0
    count_cor_pred2 = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred2[x5,y5] = 0
    count_cor_pred3 = {}
    for x5 in range(1,7):
        for y5 in range(1,7):
            count_cor_pred3[x5,y5] = 0
    
    # term_dict = {}
    # #count_course = {}
    # term_dict_predicted = {}
    # term_dict_predicted_true = {}
    # term_dict_predicted_false = {}
    # recall_bsize = {}
    # missed_bsize = {} #non-CIS courses
    # retake_bsize = {}
    # non_CIS_bsize = {}
    # CIS_missed_bsize = {}
    #rec_info = []
    #cnt_n_inst_with_future_semesters = 0
    #cnt_n_inst_with_better_recall_using_fs = 0

    #count_course_predicted = {}

    #count_one_cor_pred, count_two_cor_pred, count_three_cor_pred, count_four_cor_pred, count_five_cor_pred, count_six_or_more_cor_pred  = 0, 0, 0, 0, 0, 0
    

    #test_recall = 0.0
    num_items = len(item_dict)+1
    for i, x in enumerate(dh.batch_iter(test_data, Config().batch_size, Config().seq_len, num_items, max_basket_size, device, shuffle=False)):
    # for i, x in enumerate(dh.batch_iter(test_data, len(test_data), Config().seq_len, shuffle=False)):
        # uids, baskets, lens, prev_idx = x
        # dynamic_user, _ = dr_model(baskets, lens, hidden)
        uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = x
        #dynamic_user = model(item_seq, basket_seq, target_seq)
        basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in basket_seq]
        basket_seq_padded = pad_sequence(basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
        target_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in target_seq]
        target_seq_padded = pad_sequence(target_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_target_len]
        position_ids = torch.arange(max_basket_size).unsqueeze(0).expand(max_seq_len, -1)
        basket_ids = torch.arange(max_seq_len)
        #all_basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in baskets]
        #all_basket_seq_padded = pad_sequence(all_basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
        #dynamic_user = model(item_seq, basket_seq, target_seq)
        dynamic_user = dr_model(basket_seq_padded, position_ids, basket_ids, Config().batch_size, lens, target_seq_padded)
        #count_iter = 0
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            #dealing with last batch
            # count_iter+= 1
            # if i==39:
            #     if count_iter==12: break
            scores = []
            du_latest = du[l - 1].unsqueeze(0).to(device)
            user_baskets = test_data[test_data['userID'] == uid.item()].baskets.values[0]
            #prior_bsize = len(user_baskets)
            #print("user_baskets: ", user_baskets)

            positives = test_target[test_target['userID'] == uid.item()].baskets.values[0]  # list dim 1
            #target_semester = test_target['last_semester'][i]
            target_semester = test_data[test_data['userID'] == uid.item()].last_semester.values[0]
            positives = set(positives)
            positives = list(positives)
            p_length = len(positives)
            item_list1 = [i for i in range(num_items)]
            item_list_ten = torch.tensor(item_list1).to(device)
            #scores = list(torch.mm(du_latest, item_embedding[item_list_ten].t()).data.numpy()[0])
            scores = torch.mm(du_latest, item_embedding[item_list_ten].t()).squeeze(0)
            # item_list1= []
            # # calculating <u,p> score for all test items <u,p> pair
            # positives = test_target[test_target['userID'] == uid].baskets.values[0]  # list dim 1
            # #target_semester = test_target[test_target['userID'] == uid].last_semester.values[0]

            # #print("uid: ", uid, " ",positives)
            # for x1 in positives:
            #     item_list1.append(x1)
            # #print(positives)

            # p_length = len(positives)
            # positives2 = torch.LongTensor(positives)
            # #print(positives)
            # # Deal with positives samples
            # scores_pos = list(torch.mm(du_latest, item_embedding[positives2].t()).data.numpy()[0])
            # for s in scores_pos:
            #     scores.append(s)

            # positives_prior = [item for bskt in user_baskets for item in bskt]
            # for x3 in positives_prior:
            #     item_list1.append(x3)
            # #t_length = len(positives)
            # positives2_prior = torch.LongTensor(positives_prior)
            # scores_pos_prior = list(torch.mm(du_latest, item_embedding[positives2_prior].t()).data.numpy()[0])
            # for s in scores_pos_prior:
            #     scores.append(s)

            # # Deal with negative samples
            # #negtives = random.sample(list(neg_samples[uid]), Config().neg_num)
            # negtives = list(neg_samples[uid])
            # for x2 in negtives:
            #     item_list1.append(x2)
            # negtives2 = torch.LongTensor(negtives)
            # scores_neg = list(torch.mm(du_latest, item_embedding[negtives2].t()).data.numpy()[0])
            # for s in scores_neg:
            #     scores.append(s)
            #print(item_list1)
            #print(scores)
            # Calculate hit-ratio
            # index_k = []
            #top_k1= Config().top_k
            #top_k1 = len(positives)
            top_k1 = len(positives)
            top_k_count= 0
            top_k_count2= 0
            top_k_count3= 0
                #list_key= []
            list1= []
            list2= []
            list3 =[]
            list4= []
            list5= []
            list6 =[]
            top_k10= 10
            top_k20 = 20
            # repeat_ratio_n = round_based_on_decimal(repeat_ratio * top_k1)
            # repeat_ratio_10 = round_based_on_decimal(repeat_ratio * top_k10)
            # repeat_ratio_20 = round_based_on_decimal(repeat_ratio * top_k20)
            # new_items_ratio_n = top_k1 - repeat_ratio_n
            # new_items_ratio_10 = top_k10 - repeat_ratio_10
            # new_items_ratio_20 = top_k20 - repeat_ratio_20

            cnt_repeated_rec_n = 0
            cnt_repeated_rec_10 = 0
            cnt_repeated_rec_20 = 0
            cnt_new_item_rec_n = 0
            cnt_new_item_rec_10 = 0
            cnt_new_item_rec_20 = 0
            #print(offered_courses[l+1])
            if t_idx==1: # we are not considering randomly selected instances for last batch
                #k=0
                #pred_items= []
                count1= 0
                _, sorted_indices = torch.sort(scores, descending=True)
                list4= sorted_indices[:top_k1]
                list5= sorted_indices[:top_k10]
                list6 = sorted_indices[:top_k20] 
                list4 = list4.tolist()
                list5 = list5.tolist()
                list6 = list6.tolist()
                for index in sorted_indices:
                    #index = scores.index(max(scores))
                    # item_id1 = item_list1[index]
                    #item_id1 = index
                    item_id1 = item_list1[index.item()]
                    if item_id1 != 0 and not utils_CDREAM.filtering(item_id1, user_baskets, offered_courses[target_semester], item_dict):
                    #if not utils_tafeng_CDREAM.filtering(item_id1, user_baskets, item_dict):  # not repeated item = new item
                        if item_id1 not in list1:
                            if top_k_count<top_k1:
                                list1.append(item_id1)
                                top_k_count += 1
                                #cnt_new_item_rec_n += 1
                        if item_id1 not in list2:
                            if top_k_count2<top_k10:
                                list2.append(item_id1)
                                top_k_count2 += 1
                                #cnt_new_item_rec_10 += 1
                        if item_id1 not in list3:
                            if top_k_count3<top_k20:
                                list3.append(item_id1)
                                top_k_count3 += 1
                                #cnt_new_item_rec_20 += 1
                                    #list2.append(index_j)
                                #top_k_count+= 1
                        if(top_k1>= top_k20 and top_k_count==top_k1 and top_k_count3== top_k20): break
                        elif (top_k1< top_k20 and top_k_count3==top_k20): break
                #print(index_k)
                #print(pred_items)
                #f.write("UserID: ")
                #f.write(str(reversed_user_dict[reversed_user_dict3[uid]])+ "| ")
                #f.write(str(reversed_user_dict3[uid])+ "| ")
                #f.write("target basket: ")
                target_basket2 = []
                for item2 in positives:
                    f.write(str(reversed_item_dict[item2])+ " ")
                    target_basket2.append(reversed_item_dict[item2])

                f.write(", Recommended basket: ")
                rec_basket2 = []
                for item3 in list1:
                    f.write(str(reversed_item_dict[item3])+ " ")
                    rec_basket2.append(reversed_item_dict[item3])
                prior_courses = []
                for basket3 in user_baskets:
                    for item4 in basket3:
                        if reversed_item_dict[item4] not in prior_courses:
                            prior_courses.append(reversed_item_dict[item4])


                f.write("\n") 
                #hitratio_numer += len((set(np.arange(0, p_length)) & set(index_k)))
                hitratio_numer_n += len((set(positives) & set(list1)))
                hitratio_denom += p_length
                #print(index_k)
                pred_courses = []
                for item3 in list1:
                    pred_courses.append(reversed_item_dict[item3])
                
                #calculate recall
                recall_temp_n, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred = recall_cal_test(positives, list1, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)  
                recall_temp_10, count_at_least_one_cor_pred2, count_at_least_two_cor_pred2, count_at_least_three_cor_pred2, count_at_least_four_cor_pred2, count_at_least_five_cor_pred2, count_all_cor_pred2, count_cor_pred2 = recall_cal_test(positives, list2, count_at_least_one_cor_pred2, count_at_least_two_cor_pred2, count_at_least_three_cor_pred2, count_at_least_four_cor_pred2, count_at_least_five_cor_pred2, count_all_cor_pred2, count_cor_pred2)  
                recall_temp_20, count_at_least_one_cor_pred3, count_at_least_two_cor_pred3, count_at_least_three_cor_pred3, count_at_least_four_cor_pred3, count_at_least_five_cor_pred3, count_all_cor_pred3, count_cor_pred3 = recall_cal_test(positives, list3, count_at_least_one_cor_pred3, count_at_least_two_cor_pred3, count_at_least_three_cor_pred3, count_at_least_four_cor_pred3, count_at_least_five_cor_pred3, count_all_cor_pred3, count_cor_pred3)  
        
                recall_test_main_n += recall_temp_n
                recall_test_main_10 += recall_temp_10
                recall_test_main_20 += recall_temp_20
                ndcg_at_n = ndcg_at_k(list1, positives, p_length)
                ndcg_at_10 = ndcg_at_k(list2, positives, k=10)
                ndcg_at_20 = ndcg_at_k(list3, positives, k =20)
                sum_ndcg_at_n += ndcg_at_n
                sum_ndcg_at_10 += ndcg_at_10
                sum_ndcg_at_20 += ndcg_at_20

                recall_temp_n_wcrr, PHR_at_n_wcrr = recall_cal(positives, list4, PHR_at_n_wcrr)
                recall_temp_10_wcrr, PHR_at_10_wcrr = recall_cal(positives, list5, PHR_at_10_wcrr)
                recall_temp_20_wcrr, PHR_at_20_wcrr = recall_cal(positives, list6, PHR_at_20_wcrr)
                recall_test_main_n_wcrr += recall_temp_n_wcrr
                recall_test_main_10_wcrr += recall_temp_10_wcrr
                recall_test_main_20_wcrr += recall_temp_20_wcrr
                ndcg_at_n_wcrr = ndcg_at_k(list4, positives, p_length)
                ndcg_at_10_wcrr = ndcg_at_k(list5, positives, k=10)
                ndcg_at_20_wcrr = ndcg_at_k(list6, positives, k =20)
                sum_ndcg_at_n_wcrr += ndcg_at_n_wcrr
                sum_ndcg_at_10_wcrr += ndcg_at_10_wcrr
                sum_ndcg_at_20_wcrr += ndcg_at_20_wcrr
                #recall_2+= recall_temp
                #count+= 1
                # considering all courses taken in future semesters
                #positives = test_target[test_target['userID'] == uid].baskets.values[0]  # list dim 1
                #print(positives)

                #calculate recall considering courses taken in future semesters
                #recall_2+= recall_cal(positives, index_k)
                # recall_temp2, count_at_least_one_cor_pred2, count_at_least_two_cor_pred2, count_at_least_three_cor_pred2, count_at_least_four_cor_pred2, count_at_least_five_cor_pred2, count_all_cor_pred2, count_cor_pred2 = recall_cal(positives, top_k1, pred_items, count_at_least_one_cor_pred, count_at_least_two_cor_pred, count_at_least_three_cor_pred, count_at_least_four_cor_pred, count_at_least_five_cor_pred, count_all_cor_pred, count_cor_pred)  
                # if len(positives)> top_k1:
                #     cnt_n_inst_with_future_semesters += 1
                #     if recall_temp2>recall_temp:
                #         cnt_n_inst_with_better_recall_using_fs += 1

                
                if top_k1>=2: count_actual_bsize_at_least_2 += 1
                if top_k1>=3: count_actual_bsize_at_least_3 += 1
                if top_k1>=4: count_actual_bsize_at_least_4 += 1
                if top_k1>=5: count_actual_bsize_at_least_5 += 1
                if top_k1>=6: count_actual_bsize_at_least_6 += 1
                # target_basket2 = []
                # for item2 in positives:
                #     target_basket2.append(reversed_item_dict[item2])
                # pred_courses = []
                # for item3 in pred_items:
                #     pred_courses.append(reversed_item_dict[item3])
                # rel_rec = len((set(positives) & set(list1)))
                # row = [top_k1, target_basket2, pred_courses, rel_rec, recall_temp, target_semester]
                #rec_info.append(row)
                # test_rec_info = pd.DataFrame(rec_info, columns=['bsize', 'target_courses', 'rec_courses', 'n_rel_rec', 'recall_score', 'target_semester'])
                # test_rec_info.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_rec_info.json', orient='records', lines=True)
                # test_rec_info.to_csv('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_rec_info.csv')
                if recall_temp_n>0:  
                    recall_test_for_one_cor_pred += recall_temp_n
                correct_preds2= len((set(positives) & set(list1)))
                total_correct_preds += correct_preds2
                # if prior_bsize not in recall_bsize:
                #     recall_bsize[prior_bsize]= [recall_temp]
                # else:
                #     recall_bsize[prior_bsize] += [recall_temp]
                # # number of non-CIS and retake courses out of missed courses    
                # n_missed = 0
                # n_retake = 0
                # n_non_CIS =0
                # n_CIS =0
                # n_non_CIS_all = 0
                # unique_courses = []
                # freq = 0
                # for course2 in target_basket2:
                #     if course2 not in rec_basket2:
                #         n_missed += 1
                #         if course_CIS_dept_filtering(course2)==0:
                #             n_non_CIS +=1
                #         else:
                #             n_CIS +=1
                #         if course2 in prior_courses:
                #             n_retake += 1
                #     if course_CIS_dept_filtering(course2)==0:
                #         n_non_CIS_all += 1
                #         if course2 not in unique_courses:
                #             unique_courses.append(course2)
                #         # freq += count_course_avg_train[course2]
                #         freq += frequency_of_courses_train[course2]
                # if prior_bsize not in non_CIS_bsize:
                #     non_CIS_bsize[prior_bsize]= [n_non_CIS_all, unique_courses, 1, freq]
                # else:
                #     n3, uq, cnt3, fq = non_CIS_bsize[prior_bsize]
                #     for c4 in unique_courses:
                #         if c4 not in uq:
                #             uq.append(c4)
                #     n3 += n_non_CIS_all
                #     cnt3+= 1
                #     fq += freq
                #     non_CIS_bsize[prior_bsize] = [n3, uq, cnt3, fq]

                # if n_missed>0:
                #     if prior_bsize not in missed_bsize:
                #         missed_bsize[prior_bsize]= [n_non_CIS, n_missed]
                #     else:
                #         x3, y3 = missed_bsize[prior_bsize]
                #         x3+= n_non_CIS
                #         y3 += n_missed
                #         missed_bsize[prior_bsize] = [x3, y3]
                    
                #     if prior_bsize not in retake_bsize:
                #         retake_bsize[prior_bsize]= [n_retake, n_missed]
                #     else:
                #         x4, y4 = retake_bsize[prior_bsize]
                #         x4+= n_retake
                #         y4 += n_missed
                #         retake_bsize[prior_bsize] = [x4, y4]

                #     if prior_bsize not in CIS_missed_bsize:
                #         CIS_missed_bsize[prior_bsize]= [n_CIS, n_missed]
                #     else:
                #         x5, y5 = CIS_missed_bsize[prior_bsize]
                #         x5+= n_CIS
                #         y5 += n_missed
                #         CIS_missed_bsize[prior_bsize] = [x5, y5]
                # if n_missed>0:
                #     v3= n_non_CIS/n_missed
                #     if prior_bsize not in missed_bsize:
                #         missed_bsize[prior_bsize]= [v3]
                #     else:
                #         # x3, y3 = missed_bsize[prior_bsize]
                #         # x3+= n_non_CIS
                #         # y3 += n_missed
                #         missed_bsize[prior_bsize] += [v3]
                #     v4= n_retake/n_missed
                #     if prior_bsize not in retake_bsize:
                #         retake_bsize[prior_bsize]= [v4]
                #     else:
                #         # x4, y4 = retake_bsize[prior_bsize]
                #         # x4+= n_retake
                #         # y4 += n_missed
                #         retake_bsize[prior_bsize] += [v4]

                # if top_k1>=6: target_basket_size[6] += 1 
                # else: target_basket_size[top_k1] += 1 
                # recall_2+= recall_temp
                #course allocation for courses in target basket
                # term_dict = calculate_term_dict(term_dict, target_semester, positives, reversed_item_dict)

                # #course allocation for predicted courses
                # term_dict_predicted = calculate_term_dict(term_dict_predicted, target_semester, pred_items, reversed_item_dict)
                # term_dict_predicted_true = calculate_term_dict_true(term_dict_predicted_true, target_semester, positives, pred_items, reversed_item_dict)
                #term_dict_predicted_false = calculate_term_dict_false(term_dict_predicted_false, target_semester, positives, pred_items, reversed_item_dict)
                count=count+1
            
    hitratio_n = hitratio_numer_n / hitratio_denom
    #ndcg = ndcg / len(test_data)
    print("total count: ", count)
    #recall = recall_2/ count
    recall_test_n = recall_test_main_n/ count
    recall_test_10 = recall_test_main_10/ count
    recall_test_20 = recall_test_main_20/ count
    avg_ndcg_at_n = sum_ndcg_at_n/ count
    avg_ndcg_at_10 = sum_ndcg_at_10/ count
    avg_ndcg_at_20 = sum_ndcg_at_20/ count
    recall_test_n_wcrr = recall_test_main_n_wcrr/ count
    recall_test_10_wcrr = recall_test_main_10_wcrr/ count
    recall_test_20_wcrr = recall_test_main_20_wcrr/ count
    avg_ndcg_at_n_wcrr = sum_ndcg_at_n_wcrr/ count
    avg_ndcg_at_10_wcrr = sum_ndcg_at_10_wcrr/ count
    avg_ndcg_at_20_wcrr = sum_ndcg_at_20_wcrr/ count
    avg_PHR_at_n_wcrr = (PHR_at_n_wcrr/ count) * 100
    avg_PHR_at_10_wcrr = (PHR_at_10_wcrr/ count) * 100
    avg_PHR_at_20_wcrr = (PHR_at_20_wcrr/ count) * 100
    # print('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio))
    # f.write(str('Hit ratio[{0}]: {1}'.format(Config().top_k, hitratio)))
    print(str('Hit ratio[@n]: {0}'.format(hitratio_n)))
    f.write(str('Hit ratio[@n]: {0}'.format(hitratio_n)))
    f.write("\n")
    #print('NDCG[{0}]: {1}'.format(Config().top_k, ndcg))
    print('Recall[@n]: {0}'.format(recall_test_n))
    f.write(str('Recall[@n]: {0}'.format(recall_test_n)))
    f.write("\n")
    print("count_at_least_one_cor_pred ", count_at_least_one_cor_pred)
    f.write("count_at_least_one_cor_pred "+ str(count_at_least_one_cor_pred)+ "\n")
    percentage_of_at_least_one_cor_pred_n = (count_at_least_one_cor_pred/ len(test_target)) *100
    print("percentage_of_at_least_one_cor_pred_n: " + str(percentage_of_at_least_one_cor_pred_n)+"\n")
    f.write("percentage_of_at_least_one_cor_pred_n: " + str(percentage_of_at_least_one_cor_pred_n)+"\n")
    percentage_of_at_least_one_cor_pred_10 = (count_at_least_one_cor_pred2/ len(test_target)) *100
    print("percentage_of_at_least_one_cor_pred_10: " + str(percentage_of_at_least_one_cor_pred_10)+"\n")
    percentage_of_at_least_one_cor_pred_20 = (count_at_least_one_cor_pred3/ len(test_target)) *100
    print("percentage_of_at_least_one_cor_pred_20: " + str(percentage_of_at_least_one_cor_pred_20)+"\n")

    percentage_of_at_least_two_cor_pred_n = (count_at_least_two_cor_pred/ count_actual_bsize_at_least_2) *100
    print("percentage_of_at_least_two_cor_pred_n: ", percentage_of_at_least_two_cor_pred_n)
    f.write("percentage_of_at_least_two_cor_pred_n: "+ str(percentage_of_at_least_two_cor_pred_n)+ "\n")
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
    #calculate Recall@n for whom we generated at least one correct prediction in test data
    # test_recall_for_one_cor_pred = recall_test_for_one_cor_pred/ count_at_least_one_cor_pred
    # print("Recall@n for whom we generated at least one correct prediction in test data: ", test_recall_for_one_cor_pred)
    # f.write("Recall@n for whom we generated at least one correct prediction in test data:"+ str(test_recall_for_one_cor_pred))
    # for x6 in range(1,7):
    #     percentage_of_one_cor_pred = (count_cor_pred[x6,1]/ target_basket_size[x6]) *100
    #     print("percentage of_one cor pred for target basket size {}: {}".format(x6, percentage_of_one_cor_pred))
    #     percentage_of_two_cor_pred = (count_cor_pred[x6,2]/ target_basket_size[x6]) *100
    #     print("percentage of_two cor pred for target basket size {}: {}".format(x6, percentage_of_two_cor_pred))
    #     percentage_of_three_cor_pred = (count_cor_pred[x6,3]/ target_basket_size[x6]) *100
    #     print("percentage of_three cor pred for target basket size {}: {}".format(x6, percentage_of_three_cor_pred))
    #     percentage_of_four_cor_pred = (count_cor_pred[x6,4]/ target_basket_size[x6]) *100
    #     print("percentage of_four cor pred for target basket size {}: {}".format(x6, percentage_of_four_cor_pred))
    #     percentage_of_five_cor_pred = (count_cor_pred[x6,5]/ target_basket_size[x6]) *100
    #     print("percentage of_five cor pred for target basket size {}: {}".format(x6, percentage_of_five_cor_pred))
    #     percentage_of_at_least_six_cor_pred = (count_cor_pred[x6,6]/ target_basket_size[x6]) *100
    #     print("percentage of_at_least_six cor pred for target basket size {}: {}".format(x6, percentage_of_at_least_six_cor_pred))

    # for x7 in range(1,7):
    #     print("total count of target basket size of {}: {}".format(x7, target_basket_size[x7]))

    # for x6 in range(1,7):
    #     print("one cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,1]))
    #     print("two cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,2]))
    #     print("three cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,3]))
    #     print("four cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,4]))
    #     print("five cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,5]))
    #     print("six or more cor pred for target basket size of {}: {}".format(x6, count_cor_pred[x6,6]))
    
    # print("total correct predictions: ", total_correct_preds)
    # avg_cor_rec_per_student = (total_correct_preds/ count_at_least_one_cor_pred)
    # print("average number of courses per student correctly recommended: ", avg_cor_rec_per_student)
    # print("total number of instances with future semesters: ", cnt_n_inst_with_future_semesters)
    # print("total number of instances with better recall using future semesters: ", cnt_n_inst_with_better_recall_using_fs)

    # test_rec_info = pd.DataFrame(rec_info, columns=['bsize', 'target_courses', 'rec_courses', 'n_rel_rec', 'recall_score', 'target_semester'])

    # test_rec_info.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_rec_info.json', orient='records', lines=True)
    # test_rec_info.to_csv('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_rec_info.csv')

    # # recall scores for different basket sizes
    # recall_bsize = dict(sorted(recall_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in recall_bsize.items():
    #     bsize = k
    #     sum = 0
    #     for r in v:
    #         sum += r
    #     recall = sum/len(v)
    #     print("prior basket size: ", bsize)
    #     print("number of instances: ", len(v))
    #     print("recall score for test data: ", recall)
    
    # # number of non_CIS courses out of missed courses for different number of prior semesters
    # missed_bsize = dict(sorted(missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in missed_bsize.items():
    #     bsize = k
    #     tot_non_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     per_of_non_CIS = (tot_non_CIS/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of non CIS courses out of missed courses for test data: ", per_of_non_CIS)
    # # number of retake courses out of missed courses for different number of prior semesters
    # retake_bsize = dict(sorted(retake_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in retake_bsize.items():
    #     bsize = k
    #     tot_retake, tot_missed = v
    #     per_of_retaken_courses = (tot_retake/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print("percentage of retaken courses out of missed courses for test data: ", per_of_retaken_courses)
    
    # # calculate average nonCIS courses in test data
    # non_CIS_bsize = dict(sorted(non_CIS_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in non_CIS_bsize.items():
    #     bsize = k
    #     # sum5 = 0
    #     # for r in v:
    #     #     sum5 += r
    #     sum5, un_c, ct, freq1 = v
    #     avg_non_CIS = sum5/ct
    #     avg_pop = freq1/sum5
    #     print("prior basket size: ", bsize)
    #     print("number of instances: ", ct)
    #     print("total non_CIS_courses: ", sum5)
    #     print("average non_CIS courses for test data: ",avg_non_CIS)
    #     print("total unique non_CIS courses for test data: ",len(un_c))
    #     print("average popularity of non_CIS courses for test data: ", avg_pop)
    #     if bsize ==19: print(un_c)
    
    #  # number of CIS courses missed out of missed courses for different number of prior semesters
    # CIS_missed_bsize = dict(sorted(CIS_missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in CIS_missed_bsize.items():
    #     bsize = k
    #     tot_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     per_of_CIS_missed = (tot_CIS/ tot_missed) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of CIS courses out of missed courses: ", per_of_CIS_missed)
    

    # number of non_CIS courses out of missed courses for different number of prior semesters
    # missed_bsize = dict(sorted(missed_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in missed_bsize.items():
    #     bsize = k
    #     #tot_non_CIS, tot_missed = v
    #     # per_of_non_CIS = v[0]/ v[1]
    #     sum3 = 0
    #     for r in v:
    #         sum3 += r
    #     per_of_non_CIS = (sum3/ len(v)) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print(" percentage of non CIS courses out of missed courses for test data: ", per_of_non_CIS)
    # # number of retake courses out of missed courses for different number of prior semesters
    # retake_bsize = dict(sorted(retake_bsize.items(), key=lambda item: item[0], reverse= False))
    # for k, v in retake_bsize.items():
    #     bsize = k
    #     # tot_retake, tot_missed = v
    #     sum4 = 0
    #     for r in v:
    #         sum4 += r
    #     per_of_retaken_courses = (sum4/ len(v)) *100
    #     print("prior basket size: ", bsize)
    #     # print("number of instances: ", len(v))
    #     print("percentage of retaken courses out of missed courses for test data: ", per_of_retaken_courses)

        
    f.write("\n") 
    f.close()
    return hitratio_n, recall_test_n, recall_test_10, recall_test_20, percentage_of_at_least_one_cor_pred_n, percentage_of_at_least_one_cor_pred_10, percentage_of_at_least_one_cor_pred_20, avg_ndcg_at_n, avg_ndcg_at_10, avg_ndcg_at_20, percentage_of_at_least_two_cor_pred_n, recall_test_n_wcrr, recall_test_10_wcrr, recall_test_20_wcrr, avg_ndcg_at_n_wcrr, avg_ndcg_at_10_wcrr, avg_ndcg_at_20_wcrr, avg_PHR_at_n_wcrr, avg_PHR_at_10_wcrr, avg_PHR_at_20_wcrr


if __name__ == '__main__':
    #train()
    # train_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/train_data_all.json', orient='records', lines= True)
    # train_all, train_set_without_target, train_target,  item_dict, user_dict, reversed_item_dict, reversed_user_dict, max_len = preprocess_train_data(train_data)
    start = time.time()
    #train_data_aug = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Filtered_data/train_sample_augmented_CR.json', orient='records', lines= True)
    #train_data_unique = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/train_data_all_CR.json', orient='records', lines= True)
    train_data_unique = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_data_all_CR.json', orient='records', lines= True)
    train_data_all, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data_unique) 
    # train_all = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_sample_all.json', orient='records', lines=True)
    # train_set_without_target = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_set_without_target.json', orient='records', lines=True)
    # target_set = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/target_set.json', orient='records', lines=True)
    train_all, train_set_without_target, target_set, max_len, max_basket_size = preprocess_train_data_part2(train_data_all) 
    #valid_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/valid_data_all_CR.json', orient='records', lines= True)
    valid_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/valid_data_all_CR.json', orient='records', lines= True)
    #valid_data_excluding_summer_term = remove_summer_term_from_valid(valid_data)
    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    term_dict_train, frequency_of_courses_train, count_course_avg_train, course_sd_main, course_number_terms = calculate_avg_n_actual_courses(train_data_unique, reversed_item_dict)
    #valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data_excluding_summer_term, reversed_user_dict, item_dict)
    valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data)
    #test_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/test_data_all_CR.json', orient='records', lines= True)
    #test_data_excluding_summer_term = remove_summer_term_from_test(test_data)
    #test_data, user_dict3, reversed_user_dict3 = dataprocess.preprocess_test_data_part1(test_data_excluding_summer_term, reversed_user_dict, item_dict, reversed_user_dict2)
    test_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_data_all_CR.json', orient='records', lines= True)
    test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
    test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data)

    negative_sample(Config().NEG_SAMPLES, train_data_all, valid_set_without_target, test_set_without_target)
    offered_courses = offered_course_cal('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/all_data_CR.json')

    #offered_courses = calculate_offered_courses(train_all)
    #offered_courses = offered_course_cal('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/all_data_CR.json')
    #train(offered_courses, train_set_without_target, reversed_item_dict, reversed_user_dict)

    # dataTrain, dataTest, dataTotal, item_list, item_dict, reversed_item_dict, one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, reversed_item_dict_one_hot, user_dict_one_hot, one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train = preprocess_v30_tafeng.preprocess_data(train_data_unique)
    # num_items = one_hot_encoded_train.shape[1]
    # dataTrain, dataTest, dataTotal, item_list, item_dict, reversed_item_dict, one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, reversed_item_dict_one_hot, user_dict_one_hot, one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train, repeat_ratio2 = preprocess_v30_tafeng_ser.preprocess_data(train_data_unique)
    # num_items = one_hot_encoded_train.shape[1]
    # # doc_topic_matrix, doc_topic_df, item_embeddings, model, item_dict_idx_to_cid, item_dict_cid_to_idx = CDREAM_LGCN(df, emb_dim_LDA, max_df1, min_df1, random_state)
    # # user_embeddings_LDA, item_embeddings_LDA = create_embedding_for_training_users(one_hot_encoded_train, num_users_train, item_embeddings, item_dict_idx_to_cid, item_dict_cid_to_idx, reversed_item_dict_one_hot)
    
    # # user_embeddings_LDA2 = torch.tensor(user_embeddings_LDA, dtype=torch.float32)
    # # item_embeddings_LDA2 = torch.tensor(item_embeddings_LDA, dtype=torch.float32)
    # #df = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/processed_courses_v3.json', orient='records', lines=True)
    # #one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx, reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level = convert_side_info_to_one_hot_encoding(df, reversed_item_dict_one_hot, num_items)
    # df = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/item_side_info.json', orient='records', lines=True)
    # # ['itemID', 'PRODUCT_SUBCLASS', 'SALES_PRICE']
    # max1 = 0
    # min1 = 1000000000000
    # sum1 = 0
    # for idx1, price in enumerate(df["SALES_PRICE"]):
    #     #index = data[data['userID'] == user].index.values[0]
    #     #level_f = df["SALES_PRICE"][idx1]
    #     max1 = max(max1, int(price))
    #     min1 = min(min1, int(price))
    #     sum1 += price
    # print("max price: ", max1)
    # print("min price: ", min1)
    # print("avg price: ", sum1/len(df))
    # avg1 = int (sum1/len(df))
    # bel_avg = int (min1+ ((avg1-min1)/2))
    # abv_avg = int(avg1+((max1-avg1)/2))
    # print("bel_avg: ", bel_avg)
    # print("abv_avg: ", abv_avg)
    # dict1 = {1: 0, 2:0, 3:0}
    # for idx1, price in enumerate(df["SALES_PRICE"]):
    #     #index = data[data['userID'] == user].index.values[0]
    #     #level_f = df["SALES_PRICE"][idx1]
    #     if price>=min1 and price <= bel_avg:
    #         dict1[1] += 1
    #     elif price>=bel_avg and price <= avg1:
    #         dict1[2] += 1
    #     elif price>=avg1 and price <= max1:
    #         dict1[3] += 1
    #     # elif price>=abv_avg and price <= max1:
    #     #     dict1[4] += 1
    # print(dict1)        
    
    # one_hot_encoded_cat, one_hot_encoded_level, cat_dict_one_hot, level_dict_one_hot, reversed_dict_cat_to_idx, reversed_dict_level_to_idx, one_hot_df_cat, one_hot_df_level = convert_side_info_to_one_hot_encoding(df, reversed_item_dict_one_hot, num_items, min1, bel_avg, avg1, max1)
    # print(one_hot_encoded_cat.shape)
    # print(one_hot_encoded_level.shape)

    # cc_seq_matrix = measure_sequence_of_courses(dataTotal, reversed_item_dict_one_hot)  # course to id
    # print(cc_seq_matrix.shape)
    # threshold_weight_edges_cc= 0.2
    # # print(one_hot_encoded_cat.shape)
    # n_layers = [1,2]
    # embedding_dim = [64, 32]
    # n_epochs = [50, 100, 200, 300]
    # l_rate = [0.01, 0.001]
    # edge_dropout = [0, 0.2]
    # node_dropout = [0, 0.2]

    n_attn_layers = [2,1,3]
    attn_dropout = [0, 0.3, 0.4, 0.5]
    attn_l_rate = [0.001, 0.01]
    embedding_dim = [32, 64, 128, 16]
   # n_heads= [2,1,4]
    #embedding_dim = 64
    n_heads = [2,1,4]
    #num_layers = 2
    # max_seq_len = 86
    max_seq_len = Config().seq_len

    
    all_results = []
    cnt3 = 0
    cnt4 = 1 # 1
    # base_path = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/saved_model_LightGCN_uis_tafeng_ii_seq/'
    # #num_users = num_users_train
    # num_users = one_hot_encoded_train.shape[0]
    # num_items = one_hot_encoded_train.shape[1]
    # num_cat_f = one_hot_encoded_cat.shape[1] 
    # num_level_f = one_hot_encoded_level.shape[1]
    # # num_fet = one_hot_encoded_cat.shape[1] + one_hot_encoded_level.shape[1]
    # num_fet = num_cat_f + num_level_f
    # # device = torch.device("cpu")
    # for edge_drops in edge_dropout:
    #     for node_drops in node_dropout:
    #         for n_lays in n_layers:
    #             for emb_dim in embedding_dim:
    #                 for epoc in n_epochs:
    #                     for lr in l_rate:
    #                         #for n_hds in n_heads:
                
                                
    #                             #model, data, final_x, final_attn_weights, final_edge_index = train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, n_lays, emb_dim, epoc, lr, edge_drops, node_drops, n_hds, cnt4)
    #                             # one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, n_layers, embedding_dim, epochs, lr, edge_dropout, node_dropout, n_heads, version
                                
    #                             # 1, 128, 50, 0.2, 0, 2, 0.3, 0.01
    #                             #if (n_lays==1) and (emb_dim==64): continue
    #                             if (n_lays==1) and (emb_dim==64) and (epoc==50 or epoc==100 or epoc==200) and edge_drops==0 and node_drops==0: continue
    #                             model_filename = f"model_v{cnt4}.pth"
    #                             full_path = os.path.join(base_path, model_filename)
    #                             device = torch.device("cpu")
    #                             model = LightGCN(num_users, num_items, num_fet, emb_dim, n_lays, edge_drops, node_drops).to(device)
    #                             # GAT(num_users, num_items, num_fet, embedding_dim, n_layers, edge_dropout, node_dropout, heads)
    #                             if os.path.exists(full_path):
    #                                 print(f"Loading model from {full_path}")
    #                                 model.load_state_dict(torch.load(full_path))  # Load state_dict into the model
    #                                 item_embeddings = model.item_embeddings.weight.detach()
    #                                 # user_embeddings = model.user_embeddings.weight.detach()
    #                                 # fet_embeddings = model.fet_embeddings.weight.detach()
    #                                 # final_x = torch.cat([item_embeddings, user_embeddings, fet_embeddings], dim=0)
                                    
    #                             else:
    #                                 print(f"No saved model found at {full_path}. Training from scratch.")
    #                                 model, data, final_x = train_model(one_hot_encoded_train, one_hot_encoded_cat, one_hot_encoded_level, cc_seq_matrix, n_lays, emb_dim, epoc, lr, edge_drops, node_drops, threshold_weight_edges_cc, cnt4)
    #                                 item_embeddings = final_x[:num_items]
    #                             cnt4 += 1
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
                                # #item_embeddings = final_x[:num_items]
                                # #item_embeddings = item_embeddings.detach().numpy()
                                # num_users_train = one_hot_encoded_train.shape[0]
                                # item_embeddings_updated = update_idx_of_item_embedding(one_hot_encoded_train, num_users_train, item_embeddings, reversed_item_dict, item_dict, item_dict_one_hot) # reversed_item_dict = idx to cid
                                # #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                # item_embeddings_updated = item_embeddings_updated.to(device)
                                #item_embeddings_updated = torch.from_numpy(item_embeddings_updated).float().requires_grad_(True)
                                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                #device = torch.device("cpu")
    for attn_lr in attn_l_rate:
        for emb_dim in embedding_dim:
            for n_attn_lays in n_attn_layers:
                for attn_drops in attn_dropout:
                    for n_hds in n_heads:
                                            #if n_rnn_lays==3 and rnn_drops==0.4 and rnn_lr==0.01: continue
                                            #if n_rnn_lays==2 and rnn_drops==0.3 and rnn_lr==0.01: continue
                                            #if (epoc==50 or epoc == 100 or epoc==200) and lr==0.001: continue
                                            #if (n_lays==1) and (emb_dim==64) and (epoc==50) and (lr==0.001) and (n_rnn_lays==1 or n_rnn_lays==2): continue
                                            #if (n_lays==1) and (emb_dim==64) and (epoc==50) and (lr==0.001) and (n_rnn_lays==3) and rnn_drops==0.3: continue
                                            # if (n_lays==1) and (emb_dim==128) and (epoc==50) and (lr==0.2) and (n_rnn_lays==1): continue
                                            # if (n_lays==1) and (emb_dim==128) and (epoc==50) and (lr==0.2) and (n_rnn_lays==2) and rnn_drops==0.3: continue
                                            #if (n_lays==1) and (emb_dim==64) and (epoc==300) and lr==0.01 and edge_drops==0 and node_drops==0 and n_rnn_layers==2: continue
                                            MODEL_checkpoint, hit_ratio_n_train, recall_n_train, recall_10_train, recall_20_train, percentage_of_at_least_one_cor_pred_n_train, percentage_of_at_least_one_cor_pred_10_train, percentage_of_at_least_one_cor_pred_20_train, avg_ndcg_at_n_train, avg_ndcg_at_10_train, avg_ndcg_at_20_train = train(offered_courses, train_set_without_target, target_set, item_dict, train_data_all, valid_set_without_target, emb_dim, n_attn_lays, attn_drops, attn_lr, n_hds, max_seq_len, max_basket_size, device)
                                            MODEL_checkpoint = str(MODEL_checkpoint)
                                            end = time.time()
                                            total_training_time = end - start
                                            #print("Total training time:", total_training_time)
                                            #MODEL = "1700241865" # 64, 3, 0.6
                                            #MODEL_checkpoint = "1734321486"
                                            #MODEL_checkpoint = "1737390622"

                                            #MODEL = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Course_Beacon/runs/1674078249'

                                            while not (MODEL_checkpoint.isdigit() and len(MODEL_checkpoint) == 10):
                                                MODEL_checkpoint = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
                                            logger.info("✔︎ The format of your input is legal, now loading to next step...")

                                            MODEL_DIR = dh.load_model_file(MODEL_checkpoint)
                                            data_dir= '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/'
                                            output_dir = data_dir + "/output_dir"
                                            utils_CDREAM.create_folder(output_dir)
                                            output_path= output_dir+ "/valid_prediction_cr_38_v3.txt"
                                            hit_ratio_n_valid, recall_n_valid, recall_10_valid, recall_20_valid, percentage_of_at_least_one_cor_pred_n_valid, percentage_of_at_least_one_cor_pred_10_valid, percentage_of_at_least_one_cor_pred_20_valid, avg_ndcg_at_n_valid, avg_ndcg_at_10_valid, avg_ndcg_at_20_valid, recall_valid_n_wcrr, recall_valid_10_wcrr, recall_valid_20_wcrr, avg_ndcg_at_n_wcrr_valid, avg_ndcg_at_10_wcrr_valid, avg_ndcg_at_20_wcrr_valid, avg_PHR_at_n_wcrr_valid, avg_PHR_at_10_wcrr_valid, avg_PHR_at_20_wcrr_valid = valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, valid_set_without_target, valid_target, max_basket_size, device)
                                            output_path= output_dir+ "/test_prediction_cr_38_v3.txt"
                                            # term_dict, term_dict_predicted, term_dict_predicted_true, term_dict_predicted_false, succ_swap_test, unsucc_swap_test, no_impact_swap_test, avg_rank_swap_test, std_rank_swap_test, cnt_swap_in_n_inst_test, cnt_tot_swap_test, avg_recall_swap_test, avg_recall_no_swap_test, tot_courses_test, cnt_succ_swaps_num_inst, cnt_unsucc_swaps_num_inst, cnt_no_impact_swaps_num_inst, cnt_both_succ_unsucc_swaps_num_inst, cnt_exact_1_swap_num_inst, cnt_exact_2_swaps_num_inst, cnt_at_least_3_swaps_num_inst, cnt_at_least_one_unsuc_and_at_least_two_swaps, cnt_at_least_two_unsuc_and_at_least_two_swaps = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, output_path, course_prereq)
                                            hit_ratio_n_test, recall_n_test, recall_10_test, recall_20_test, percentage_of_at_least_one_cor_pred_n_test, percentage_of_at_least_one_cor_pred_10_test, percentage_of_at_least_one_cor_pred_20_test, avg_ndcg_at_n_test, avg_ndcg_at_10_test, avg_ndcg_at_20_test, percentage_of_at_least_two_cor_pred_n_test, recall_test_n_wcrr, recall_test_10_wcrr, recall_test_20_wcrr, avg_ndcg_at_n_wcrr_test, avg_ndcg_at_10_wcrr_test, avg_ndcg_at_20_wcrr_test, avg_PHR_at_n_wcrr_test, avg_PHR_at_10_wcrr_test, avg_PHR_at_20_wcrr_test = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_set_without_target, test_target, max_basket_size, device)
                                            # print("test recall@n: ", test_recall)
                                            # print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
                                            # print("percentage_of_at_least_two_cor_pred: ", percentage_of_at_least_two_cor_pred)
                                            # print("validation recall@n: ", valid_recall)
                                            # # print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred2)
                                            # print("train recall@n: ", train_recall)
                                            print("test recall@n: ", recall_n_test)
                                            print("test recall@10: ", recall_10_test)
                                            print("test recall@20: ", recall_20_test)
                                            print("percentage_of_at_least_one_cor_pred test@n: ", percentage_of_at_least_one_cor_pred_n_test)
                                            print("percentage_of_at_least_one_cor_pred test@10: ", percentage_of_at_least_one_cor_pred_10_test)
                                            print("percentage_of_at_least_one_cor_pred test@20: ", percentage_of_at_least_one_cor_pred_20_test)
                                            print("percentage_of_at_least_two_cor_pred test @n: ", percentage_of_at_least_two_cor_pred_n_test)
                                            print("validation recall@n: ", recall_n_valid)
                                            print("validation recall@10: ", recall_10_valid)
                                            print("validation recall@20: ", recall_20_valid)
                                            print("percentage_of_at_least_one_cor_pred valid@n: ", percentage_of_at_least_one_cor_pred_n_valid)
                                            print("percentage_of_at_least_one_cor_pred valid@10: ", percentage_of_at_least_one_cor_pred_10_valid)
                                            print("percentage_of_at_least_one_cor_pred valid @20: ", percentage_of_at_least_one_cor_pred_20_valid)
                                            #print("percentage_of_at_least_two_cor_pred valid @n: ", percentage_of_at_least_two_cor_pred_n_valid)

                                            # print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred2)
                                            print("train recall@n: ", recall_n_train)
                                            print("train recall@10: ", recall_10_train)
                                            print("train recall@20: ", recall_20_train)

                                            print("test recall@n wcrr: ", recall_test_n_wcrr)
                                            print("test recall@10 wcrr: ", recall_test_10_wcrr)
                                            print("test recall@20 wcrr: ", recall_test_20_wcrr)
                                            print("valid recall@n wcrr: ", recall_valid_n_wcrr)
                                            print("valid recall@10 wcrr: ", recall_valid_10_wcrr)
                                            print("valid recall@20 wcrr: ", recall_valid_20_wcrr)
                                            # print("percentage_of_at_least_one_cor_pred: ", percentage_of_at_least_one_cor_pred)
                                            # print("number of components: ", num_components)
                                            # end4 = time.time()
                                            # print("time for recommendation for test data:", end4-end3)
                                            # print(f"n_layers: {n_layers}, \n embedding_dim: {embedding_dim}") 
                                            # print("n_epochs: ", n_epochs)
                                            # print("n_heads: ", n_heads)
                                            # print("l_rate: ", l_rate)
                                            # print("dropout: ", dropout)
                                            # #print("shape of x for test: ", updated_data_test.x.shape)
                                            # print("n_layers: ", n_lays)
                                            print("emb_dim: ", emb_dim)
                                            # print("epochs: ", epoc)
                                            # print("learning rate: ", lr)
                                            print("n of heads", n_hds)
                                            # print("edge dropout: ", edge_drops)
                                            # print("node dropout: ", node_drops)
                                            cnt3 += 1
                                            print("cnt of iteration: ", cnt3)
                                            print( "num_of_attn_layers: ", n_attn_lays)
                                            print("dropout in attn: ",  attn_drops)
                                            print("lr in attn: ",  attn_lr)
                                            row1 = [emb_dim, n_attn_lays, attn_drops, attn_lr, n_hds, recall_n_train, recall_n_valid, recall_n_test, recall_10_train, recall_10_valid, recall_10_test, recall_20_train, recall_20_valid, recall_20_test, avg_ndcg_at_n_valid, avg_ndcg_at_n_test, avg_ndcg_at_10_valid, avg_ndcg_at_10_test, avg_ndcg_at_20_valid, avg_ndcg_at_20_test, percentage_of_at_least_one_cor_pred_n_valid, percentage_of_at_least_one_cor_pred_n_test, percentage_of_at_least_one_cor_pred_10_valid, percentage_of_at_least_one_cor_pred_10_test, percentage_of_at_least_one_cor_pred_20_valid, percentage_of_at_least_one_cor_pred_20_test, percentage_of_at_least_two_cor_pred_n_test, recall_valid_n_wcrr, recall_test_n_wcrr, recall_valid_10_wcrr, recall_test_10_wcrr, recall_valid_20_wcrr, recall_test_20_wcrr, avg_ndcg_at_n_wcrr_valid, avg_ndcg_at_n_wcrr_test, avg_ndcg_at_10_wcrr_valid, avg_ndcg_at_10_wcrr_test, avg_ndcg_at_20_wcrr_valid, avg_ndcg_at_20_wcrr_test, avg_PHR_at_n_wcrr_valid, avg_PHR_at_n_wcrr_test, avg_PHR_at_10_wcrr_valid, avg_PHR_at_10_wcrr_test, avg_PHR_at_20_wcrr_valid, avg_PHR_at_20_wcrr_test]
                                            all_results.append(row1)
                                            all_results_df = pd.DataFrame(all_results, columns=['emb_dim', 'attn_layer_number', 'attn_dropout', 'attn_l_rate', 'n_heads', 'recall_n_train', 'recall_n_valid', 'recall_n_test', 'recall_10_train', 'recall_10_valid', 'recall_10_test', 'recall_20_train', 'recall_20_valid', 'recall_20_test', 'valid_ndcg@n', 'test_ndcg@n', 'valid_ndcg@10', 'test_ndcg@10', 'valid_ndcg@20', 'test_ndcg@20', 'PHR@n_valid', 'PHR@n_test', 'PHR@10_valid', 'PHR@10_test', 'PHR@20_valid', 'PHR@20_test', 'percentage_of_at_least_two_cor_pred_n_test', 'recall_valid_n_wcrr', 'recall_test_n_wcrr', 'recall_valid_10_wcrr', 'recall_test_10_wcrr', 'recall_valid_20_wcrr', 'recall_test_20_wcrr', 'avg_ndcg_at_n_wcrr_valid', 'avg_ndcg_at_n_wcrr_test', 'avg_ndcg_at_10_wcrr_valid', 'avg_ndcg_at_10_wcrr_test', 'avg_ndcg_at_20_wcrr_valid', 'avg_ndcg_at_20_wcrr_test', 'avg_PHR_at_n_wcrr_valid', 'avg_PHR_at_n_wcrr_test', 'avg_PHR_at_10_wcrr_valid', 'avg_PHR_at_10_wcrr_test', 'avg_PHR_at_20_wcrr_valid', 'avg_PHR_at_20_wcrr_test'])
                                            
                                            all_results_df.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/all_results_df_GenRec_updated_cr_v100_GPU_v3.json', orient='records', lines=True) 
                                            all_results_df.to_csv('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/all_results_df_GenRec_updated_cr_v100_GPU_v3.csv') 
                                            # quartz
                                            # update score calculation
                                            # running (8) again
                                            # running 
                                           
                                            
