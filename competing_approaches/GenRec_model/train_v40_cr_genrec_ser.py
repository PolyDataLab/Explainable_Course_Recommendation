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
import data_helpers_genrec as dh
from config_genrec import Config # type: ignore
from genrec_model import GenRec
import tensorflow as tf
from dataprocess_genrec import *
#from utils import *
import utils # type: ignore
from utils import * # type: ignore
#from offered_courses import *
import pandas as pd
import math
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

    except KeyboardInterrupt:
        logger.info('*' * 89)
        logger.info('Early Stopping!')
    print("model directory: ", timestamp)
    #print("config for train: 64, 2, 0.6")

    return timestamp, hit_ratio_n, recall_test_n, recall_test_10, recall_test_20, percentage_of_at_least_one_cor_pred_n, percentage_of_at_least_one_cor_pred_10, percentage_of_at_least_one_cor_pred_20, avg_ndcg_at_n, avg_ndcg_at_10, avg_ndcg_at_20

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
    # valid_data = dh.load_data('./DataSet/University_data/valid_sample_without_target.json')

    logger.info("✔︎ Test data processing...")
    #test_target = dh.load_data(Config().TESTSET_DIR)
    #valid_target = dh.load_data('./DataSet/University_data/validation_target_set.json')

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    dr_model = torch.load(MODEL_DIR)
    dr_model = dr_model.to(device)  # Move to the appropriate devic

    dr_model.eval()

    item_embedding = dr_model.item_embedding.weight
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
               
                count+= 1
                
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
# testing with GenRec model
def test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_data, test_target, max_basket_size, device):
    f = open(output_path, "w") #generating text file with recommendation using filtering function

    logger.info("✔︎ Load negative sample...")
    with open(Config().NEG_SAMPLES, 'rb') as handle:
        neg_samples = pickle.load(handle)

    # Load model
    dr_model = torch.load(MODEL_DIR)
    dr_model = dr_model.to(device)  # Move to the appropriate device

    dr_model.eval()

    item_embedding = dr_model.item_embedding.weight
    hitratio_numer_n = 0
    hitratio_denom = 0
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
            
    num_items = len(item_dict)+1
    for i, x in enumerate(dh.batch_iter(test_data, Config().batch_size, Config().seq_len, num_items, max_basket_size, device, shuffle=False)):
    # for i, x in enumerate(dh.batch_iter(test_data, len(test_data), Config().seq_len, shuffle=False)):
        
        uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = x
        #dynamic_user = model(item_seq, basket_seq, target_seq)
        basket_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in basket_seq]
        basket_seq_padded = pad_sequence(basket_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_basket_len]
        target_seq_tensors = [torch.tensor(user, dtype=torch.long, device=device) for user in target_seq]
        target_seq_padded = pad_sequence(target_seq_tensors, batch_first=True, padding_value=0)  # [batch_size, max_target_len]
        position_ids = torch.arange(max_basket_size).unsqueeze(0).expand(max_seq_len, -1)
        basket_ids = torch.arange(max_seq_len)
        dynamic_user = dr_model(basket_seq_padded, position_ids, basket_ids, Config().batch_size, lens, target_seq_padded)
        #count_iter = 0
        for uid, l, du, t_idx in zip(uids, lens, dynamic_user, prev_idx):
            
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
            
            top_k1 = len(positives)
            top_k_count= 0
            top_k_count2= 0
            top_k_count3= 0
        
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
                    item_id1 = item_list1[index.item()]
                    if item_id1 != 0 and not utils.filtering(item_id1, user_baskets, offered_courses[target_semester], item_dict): # removing prior courses taken by a student and a course not offered in the target semester
                   
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

                # recall_temp_n_wcrr, PHR_at_n_wcrr = recall_cal(positives, list4, PHR_at_n_wcrr)
                # recall_temp_10_wcrr, PHR_at_10_wcrr = recall_cal(positives, list5, PHR_at_10_wcrr)
                # recall_temp_20_wcrr, PHR_at_20_wcrr = recall_cal(positives, list6, PHR_at_20_wcrr)
                # recall_test_main_n_wcrr += recall_temp_n_wcrr
                # recall_test_main_10_wcrr += recall_temp_10_wcrr
                # recall_test_main_20_wcrr += recall_temp_20_wcrr
                # ndcg_at_n_wcrr = ndcg_at_k(list4, positives, p_length)
                # ndcg_at_10_wcrr = ndcg_at_k(list5, positives, k=10)
                # ndcg_at_20_wcrr = ndcg_at_k(list6, positives, k =20)
                # sum_ndcg_at_n_wcrr += ndcg_at_n_wcrr
                # sum_ndcg_at_10_wcrr += ndcg_at_10_wcrr
                # sum_ndcg_at_20_wcrr += ndcg_at_20_wcrr
                #recall_2+= recall_temp
                #count+= 1

                if top_k1>=2: count_actual_bsize_at_least_2 += 1
                if top_k1>=3: count_actual_bsize_at_least_3 += 1
                if top_k1>=4: count_actual_bsize_at_least_4 += 1
                if top_k1>=5: count_actual_bsize_at_least_5 += 1
                if top_k1>=6: count_actual_bsize_at_least_6 += 1
    
                if recall_temp_n>0:  
                    recall_test_for_one_cor_pred += recall_temp_n
                correct_preds2= len((set(positives) & set(list1)))
                total_correct_preds += correct_preds2
                count=count+1 # counting number of test instances
            
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
    # recall_test_n_wcrr = recall_test_main_n_wcrr/ count
    # recall_test_10_wcrr = recall_test_main_10_wcrr/ count
    # recall_test_20_wcrr = recall_test_main_20_wcrr/ count
    # avg_ndcg_at_n_wcrr = sum_ndcg_at_n_wcrr/ count
    # avg_ndcg_at_10_wcrr = sum_ndcg_at_10_wcrr/ count
    # avg_ndcg_at_20_wcrr = sum_ndcg_at_20_wcrr/ count
    # avg_PHR_at_n_wcrr = (PHR_at_n_wcrr/ count) * 100
    # avg_PHR_at_10_wcrr = (PHR_at_10_wcrr/ count) * 100
    # avg_PHR_at_20_wcrr = (PHR_at_20_wcrr/ count) * 100
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
        
    f.write("\n") 
    f.close()
    return hitratio_n, recall_test_n, recall_test_10, recall_test_20, percentage_of_at_least_one_cor_pred_n, percentage_of_at_least_one_cor_pred_10, percentage_of_at_least_one_cor_pred_20, avg_ndcg_at_n, avg_ndcg_at_10, avg_ndcg_at_20, percentage_of_at_least_two_cor_pred_n, recall_test_n_wcrr, recall_test_10_wcrr, recall_test_20_wcrr, avg_ndcg_at_n_wcrr, avg_ndcg_at_10_wcrr, avg_ndcg_at_20_wcrr, avg_PHR_at_n_wcrr, avg_PHR_at_10_wcrr, avg_PHR_at_20_wcrr


if __name__ == '__main__':
    
    start = time.time()
    train_data_unique = pd.read_json('./DataSet/University_data/train_data_all.json', orient='records', lines= True)
    train_data_all, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data_unique) 
    train_all, train_set_without_target, target_set, max_len, max_basket_size = preprocess_train_data_part2(train_data_all) 
    
    valid_data = pd.read_json('./DataSet/University_data/valid_data_all.json', orient='records', lines= True)
    valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
    valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data)
    test_data = pd.read_json('./DataSet/University_data/test_data_all_CR.json', orient='records', lines= True)
    test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
    test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data)

    negative_sample(Config().NEG_SAMPLES, train_data_all, valid_set_without_target, test_set_without_target)
    offered_courses = offered_course_cal('./Dataset/University_data/all_data.json')

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    for attn_lr in attn_l_rate:
        for emb_dim in embedding_dim:
            for n_attn_lays in n_attn_layers:
                for attn_drops in attn_dropout:
                    for n_hds in n_heads:
                                            MODEL_checkpoint, hit_ratio_n_train, recall_n_train, recall_10_train, recall_20_train, percentage_of_at_least_one_cor_pred_n_train, percentage_of_at_least_one_cor_pred_10_train, percentage_of_at_least_one_cor_pred_20_train, avg_ndcg_at_n_train, avg_ndcg_at_10_train, avg_ndcg_at_20_train = train(offered_courses, train_set_without_target, target_set, item_dict, train_data_all, valid_set_without_target, emb_dim, n_attn_lays, attn_drops, attn_lr, n_hds, max_seq_len, max_basket_size, device)
                                            MODEL_checkpoint = str(MODEL_checkpoint)
                                            end = time.time()
                                            total_training_time = end - start
                                            #print("Total training time:", total_training_time)

                                            while not (MODEL_checkpoint.isdigit() and len(MODEL_checkpoint) == 10):
                                                MODEL_checkpoint = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
                                            logger.info("✔︎ The format of your input is legal, now loading to next step...")

                                            MODEL_DIR = dh.load_model_file(MODEL_checkpoint)
                                            data_dir= './DataSet/University_data/'
                                            output_dir = data_dir + "/output_dir"
                                            utils_CDREAM.create_folder(output_dir)
                                            output_path= output_dir+ "/valid_prediction_cr_38_v3.txt"
                                            hit_ratio_n_valid, recall_n_valid, recall_10_valid, recall_20_valid, percentage_of_at_least_one_cor_pred_n_valid, percentage_of_at_least_one_cor_pred_10_valid, percentage_of_at_least_one_cor_pred_20_valid, avg_ndcg_at_n_valid, avg_ndcg_at_10_valid, avg_ndcg_at_20_valid, recall_valid_n_wcrr, recall_valid_10_wcrr, recall_valid_20_wcrr, avg_ndcg_at_n_wcrr_valid, avg_ndcg_at_10_wcrr_valid, avg_ndcg_at_20_wcrr_valid, avg_PHR_at_n_wcrr_valid, avg_PHR_at_10_wcrr_valid, avg_PHR_at_20_wcrr_valid = valid(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict2, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, valid_set_without_target, valid_target, max_basket_size, device)
                                            output_path= output_dir+ "/test_prediction_cr_38_v3.txt"
                                            hit_ratio_n_test, recall_n_test, recall_10_test, recall_20_test, percentage_of_at_least_one_cor_pred_n_test, percentage_of_at_least_one_cor_pred_10_test, percentage_of_at_least_one_cor_pred_20_test, avg_ndcg_at_n_test, avg_ndcg_at_10_test, avg_ndcg_at_20_test, percentage_of_at_least_two_cor_pred_n_test, recall_test_n_wcrr, recall_test_10_wcrr, recall_test_20_wcrr, avg_ndcg_at_n_wcrr_test, avg_ndcg_at_10_wcrr_test, avg_ndcg_at_20_wcrr_test, avg_PHR_at_n_wcrr_test, avg_PHR_at_10_wcrr_test, avg_PHR_at_20_wcrr_test = test(offered_courses, reversed_item_dict, reversed_user_dict, item_dict, reversed_user_dict3, frequency_of_courses_train, count_course_avg_train, MODEL_DIR, output_path, test_set_without_target, test_target, max_basket_size, device)
                                    
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

                                            # print("test recall@n wcrr: ", recall_test_n_wcrr)
                                            # print("test recall@10 wcrr: ", recall_test_10_wcrr)
                                            # print("test recall@20 wcrr: ", recall_test_20_wcrr)
                                            # print("valid recall@n wcrr: ", recall_valid_n_wcrr)
                                            # print("valid recall@10 wcrr: ", recall_valid_10_wcrr)
                                            # print("valid recall@20 wcrr: ", recall_valid_20_wcrr)
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
                                            
                                            all_results_df.to_json('./GenRec_model/all_results_df_GenRec_v3.json', orient='records', lines=True) 
                                            all_results_df.to_csv('./GenRec_model/all_results_df_GenRec_v3.csv') 
                                        
                                           
                                            
