import os
import logging
import torch
import numpy as np
import pandas as pd

#logger function for logging information
def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger

#loading data
def load_data(input_file, flag=None):
    if flag:
        data = pd.read_json(input_file, orient='records', lines=True)
        #data = pd.read_csv(input_file, orient='records', lines=True)
    else:
        data = pd.read_json(input_file, orient='records', lines=True)
        #data = pd.read_csv(input_file, orient='records', lines=True)

    return data

#loading model file
def load_model_file(checkpoint_dir):
    MODEL_DIR = './MMNR/runs_cr_genrec_v1/' + checkpoint_dir
    #MODEL_DIR = checkpoint_dir
    names = [name for name in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, name))]
    max_epoch = 0
    choose_model = ''
    for name in names:
        if int(name[6:8]) >= max_epoch:
            max_epoch = int(name[6:8])
            choose_model = name
    MODEL_FILE = './MMNR/runs_cr_genrec_v1/' + checkpoint_dir + '/' + choose_model
    #MODEL_FILE = checkpoint_dir + '/' + choose_model
    return MODEL_FILE


def sort_batch_of_lists(uids, item_seq, basket_seq, target_seq, baskets, lens, device):
    """Sort batch of lists according to len(list). Descending"""
    sorted_idx = [i[0] for i in sorted(enumerate(lens), key=lambda x: x[1], reverse=True)]
    uids = [uids[i] for i in sorted_idx]
    lens = [lens[i] for i in sorted_idx]
    # batch_of_lists = [batch_of_lists[i] for i in sorted_idx]
    item_seq = [item_seq[i] for i in sorted_idx]
    basket_seq = [basket_seq[i] for i in sorted_idx]
    target_seq = [target_seq[i] for i in sorted_idx]
    baskets = [baskets[i] for i in sorted_idx]
    prev_idx = []
    for idx in sorted_idx:
       prev_idx.append(1)
    return uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx

def sort_batch_of_lists_2(uids, item_seq, basket_seq, target_seq, lens, last_batch_actual_size, device):
    """Sort batch of lists according to len(list). Descending"""
    sorted_idx = [i[0] for i in sorted(enumerate(lens), key=lambda x: x[1], reverse=True)]
    uids = [uids[i] for i in sorted_idx]
    lens = [lens[i] for i in sorted_idx]
    #batch_of_lists = [batch_of_lists[i] for i in sorted_idx]
    item_seq = [item_seq[i] for i in sorted_idx]
    basket_seq = [basket_seq[i] for i in sorted_idx]
    target_seq = [target_seq[i] for i in sorted_idx]
    prev_idx = []
    for idx in sorted_idx:
        if(idx<last_batch_actual_size):
            prev_idx.append(1)
        else:
            prev_idx.append(0) #randomly taken to maintain fixed length batch size

    return uids, item_seq, basket_seq, target_seq, lens, prev_idx



def pad_batch_of_lists(batch_of_lists, pad_len, device):
    """Pad batch of lists."""
    padded = [l + [[0]] * (pad_len - len(l)) for l in batch_of_lists]
    return padded
def pad_batch_of_lists2(batch_of_lists, pad_len, device):
    """Pad batch of lists."""
    padded = [l + [0] * (pad_len - len(l)) for l in batch_of_lists]
    padded = torch.tensor(padded, device = device)
    return padded
def pad_batch_of_lists3(batch_of_lists, pad_len_item, device):
    """Pad batch of lists."""
    padded = [l + [0] * (pad_len_item - len(l)) for l in batch_of_lists]
    return padded
def pad_batch_of_lists4(batch_of_lists, pad_len, pad_len_item, device):
    """Pad batch of lists."""
    updated_batch_of_lists = []
    for bskts in batch_of_lists:
        bskts2= []
        for bskt in bskts:
            if len(bskt)>=pad_len_item:
                padded1= bskt[:pad_len_item]
            else:
                padded1 = bskt + [0] * (pad_len_item - len(bskt))
            bskts2.append(padded1)
        padded2 = bskts2 + [[0]* pad_len_item] * (pad_len - len(bskts2))
        updated_batch_of_lists.append(padded2)

    #padded = [l + [[0]] * (pad_len - len(l)) for l in batch_of_lists]
    return updated_batch_of_lists

def batch_iter(data, batch_size, pad_len, pad_len_item, max_basket_size, device, shuffle=True):
    """
    Turn dataset into iterable batches.

    Args:
        data: The data
        batch_size: The size of the data batch
        pad_len: The padding length
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size
    if shuffle:
        shuffled_data = data.sample(frac=1)
    else:
        shuffled_data = data

    for i in range(num_batches_per_epoch):
        #print(shuffled_data)
        uids = torch.tensor(shuffled_data.iloc[i * batch_size: (i + 1) * batch_size].userID.values, device=device)
        baskets = list(shuffled_data.iloc[i * batch_size: (i + 1) * batch_size].baskets.values)
        basket_seq = []
        item_seq = []
        target_seq = []
        for baskets2 in baskets:
            train_basket = baskets2[0:-1]
            target_basket = baskets2[-1]
            user_item_seq = [item for basket in train_basket for item in basket]
            basket_seq.append(train_basket)
            item_seq.append(user_item_seq)
            target_seq.append(target_basket)

        lens = torch.tensor(shuffled_data.iloc[i * batch_size: (i + 1) * batch_size].num_baskets.values, device=device) # subtracting 1 (last basket)
        #uids = shuffled_data[i * batch_size: (i + 1) * batch_size].userID.values
        #baskets = list(shuffled_data[i * batch_size: (i + 1) * batch_size].baskets.values)
        #lens = shuffled_data[i * batch_size: (i + 1) * batch_size].num_baskets.values
        # uids, baskets, lens, prev_idx = sort_batch_of_lists(uids, baskets, lens, device)  
        uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = sort_batch_of_lists(uids, item_seq, basket_seq, target_seq, baskets, lens, device) 
        # baskets = pad_batch_of_lists(baskets, pad_len, device)
        baskets = pad_batch_of_lists4(baskets, pad_len, max_basket_size, device)
        item_seq = pad_batch_of_lists3(item_seq, pad_len_item, device)
        # basket_seq = pad_batch_of_lists(basket_seq, pad_len, device)
        basket_seq = pad_batch_of_lists4(basket_seq, pad_len, max_basket_size, device)
        target_seq = pad_batch_of_lists3(target_seq, max_basket_size, device)
        # yield uids, baskets, lens, prev_idx
        yield uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx

    if data_size % batch_size != 0:
        residual = [i for i in range(num_batches_per_epoch * batch_size, data_size)] + list(
            np.random.choice(data_size, batch_size - data_size % batch_size))
        # uids = shuffled_data.iloc[residual].userID.values
        # baskets = list(shuffled_data.iloc[residual].baskets.values)
        # lens = shuffled_data.iloc[residual].num_baskets.values
        uids = torch.tensor(shuffled_data.iloc[residual].userID.values, device=device)
        baskets = list(shuffled_data.iloc[residual].baskets.values)
        basket_seq = []
        item_seq = []
        target_seq = []
        for baskets2 in baskets:
            train_basket = baskets2[0:-1]
            target_basket = baskets2[-1]
            user_item_seq = [item for basket in train_basket for item in basket]
            basket_seq.append(train_basket)
            item_seq.append(user_item_seq)
            target_seq.append(target_basket)
        lens = torch.tensor(shuffled_data.iloc[residual].num_baskets.values, device=device)
        # uids, baskets, lens, prev_idx = sort_batch_of_lists_2(uids, baskets, lens, data_size % batch_size, device)
        uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx = sort_batch_of_lists(uids, item_seq, basket_seq, target_seq, baskets, lens, device) 
        baskets = pad_batch_of_lists4(baskets, pad_len, max_basket_size, device)
        item_seq = pad_batch_of_lists3(item_seq, pad_len_item, device)
        # basket_seq = pad_batch_of_lists(basket_seq, pad_len, device)
        basket_seq = pad_batch_of_lists4(basket_seq, pad_len, max_basket_size, device)
        target_seq = pad_batch_of_lists3(target_seq, max_basket_size, device)
        # yield uids, baskets, lens, prev_idx
        yield uids, item_seq, basket_seq, target_seq, baskets, lens, prev_idx


def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]


def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)
