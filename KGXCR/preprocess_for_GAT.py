# preprocess for adding user nodes (one at a time or all nodes together) or use updated embeddings without fitting a new graph into the model
# for diamond server
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
#from sklearn.model_selection import train_test_split
import json
import random
#from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def convert_to_one_hot_encoding(data):
    # One-hot encode the 'interactions' column
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(data['interacted_items'])
    # Convert back to a DataFrame for easier readability
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_, index=data['userID']).reset_index(drop=True)
    item_dict_one_hot = {index: item_id for index, item_id in enumerate(one_hot_df.columns)}
    user_dict_one_hot = {index: user_id for index, user_id in enumerate(data['userID'])}
    one_hot_df['userID'] = data['userID']
    list1 =  list(item_dict_one_hot.values())
    one_hot_df = one_hot_df[['userID'] + list1]
    #print(one_hot_df.shape)
    
    return one_hot_encoded, one_hot_df, item_dict_one_hot, user_dict_one_hot

def convert_to_one_hot_encoding_valid_test(test_data_interactions, reversed_item_dict_one_hot, num_users_train):
    num_users_test = len(test_data_interactions)
    num_items_train = len(reversed_item_dict_one_hot)
    user_dict_one_hot = {index: user_id for index, user_id in enumerate(test_data_interactions['userID'])}
    reversed_user_dict_one_hot = {user_id: index for index, user_id in enumerate(test_data_interactions['userID'])}

    # Initialize a 2D array of zeros with shape (num_users_test, num_items_train)
    test_user_item_array = np.zeros((num_users_test, num_items_train), dtype=int)

    # Fill the array using test interactions and item_dict
    for idx in range(len(test_data_interactions)):
        userID = test_data_interactions['userID'][idx]
        itemIDs = test_data_interactions['interacted_items'][idx]
        for item_id in itemIDs:
            if item_id in reversed_item_dict_one_hot:  # Only add items found in the training item_dict
                item_index = reversed_item_dict_one_hot[item_id]
                test_user_item_array[idx, item_index] = 1
    return test_user_item_array, user_dict_one_hot, reversed_user_dict_one_hot

def preprocess_data(input_data):
    
    data = input_data
    
    itemIDs = {}
    index=0
    for baskets in data['baskets']:
        for basket in baskets:
            for item in basket:
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)
        index +=1   
    # baskets['num_baskets'] = baskets.baskets.apply(len)
    item_list = list(itemIDs.keys())
    reversed_item_dict = dict(zip(itemIDs.values(), itemIDs.keys()))
    users = data.userID.values
    #training set
    count1 = 0
    train_valid = []
    train_interactions = []
    index = 0
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            #max_len = max(max_len, data.iloc[index]['num_baskets'])
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_valid.append(row)
            item_ids = []
            for basket in b:
                for item in basket:
                    if item not in item_ids:
                        item_ids.append(item)
            row1 = [user, item_ids]
            train_interactions.append(row1)
            count1 += 1

        #if count1==100: break
        
    total_set = pd.DataFrame(train_valid, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])
    total_interactions = pd.DataFrame(train_interactions, columns=['userID', 'interacted_items'])
    one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, user_dict_one_hot = convert_to_one_hot_encoding(total_interactions)
    reversed_item_dict_one_hot = dict(zip(item_dict_one_hot.values(), item_dict_one_hot.keys()))  # course, index
    train_valid2 = []
    train_interactions_without_target = []
    count1 = 0
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']-1, data.iloc[index]['last_semester']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            train_valid2.append(row)
            item_ids = []
            for basket in b:
                for item in basket:
                    if item not in item_ids:
                        item_ids.append(item)
            row1 = [user, item_ids]
            train_interactions_without_target.append(row1)
            count1 += 1
            #if count1==100: break
        
    train_set = pd.DataFrame(train_valid2, columns=['userID', 'baskets', 'num_baskets', 'last_semester'])
    train_interactions_without_target1 = pd.DataFrame(train_interactions_without_target, columns=['userID', 'interacted_items'])
    num_users_train = len(user_dict_one_hot)
    one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train  = convert_to_one_hot_encoding_valid_test(train_interactions_without_target1, reversed_item_dict_one_hot, num_users_train)
    target_set = []
    count1= 0
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][-1]
        if data.iloc[index]['num_baskets']>=3:
            b1 = [b]
            row = [user, b1, data.iloc[index]['last_semester']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            target_set.append(row)
            count1 += 1
            #if count1==100: break
    
    target_set = pd.DataFrame(target_set, columns=['userID', 'baskets', 'last_semester'])
    
    train_set.to_json('./train_sample_updated_v3.json', orient='records', lines=True)
    target_set.to_json('./target_set_v3.json', orient='records', lines=True)
    total_set.to_json('./total_sample_updated_v3.json', orient='records', lines=True)
    one_hot_encoded_df_train.to_json('./one_hot_encoded_df_train.json', orient='records', lines=True)
    one_hot_encoded_df_train.to_csv('./one_hot_encoded_df_train.csv')
    
    print("done!")
    #print("processing took {0:.1f} sec".format(time.time() - start))
    return train_set, target_set, total_set, item_list, itemIDs, reversed_item_dict, one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, reversed_item_dict_one_hot, user_dict_one_hot, one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train

def preprocess_valid_data(input_data, item_list, reversed_item_dict_one_hot, num_users_train): #  
  
    data = input_data
   

    #baskets['num_baskets'] = baskets.baskets.apply(len)
    index=0
    for baskets in data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(data['timestamps'][index][tsindex])
            tsindex += 1
        data['baskets'][index] = new_baskets
        data['num_baskets'][index] = len(new_baskets)
        data['timestamps'][index] = ts
        index +=1  
    users = data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
       
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps'] ]
            test_all.append(row)
        index +=1
        #if index==30: break
    valid_set_all = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])

    test_2 = []
    index = 0
    valid_interactions_without_target = []
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']-1, data.iloc[index]['last_semester']]
            test_2.append(row)
            item_ids = []
            for basket in b:
                for item in basket:
                    if item not in item_ids:
                        item_ids.append(item)
            row1 = [user, item_ids]
            valid_interactions_without_target.append(row1)
        index +=1
        #if index==30: break
   
    valid_set_without_target = pd.DataFrame(test_2, columns=['userID', 'baskets', 'num_baskets', 'last_semester'])
    valid_interactions_without_target1 = pd.DataFrame(valid_interactions_without_target, columns=['userID', 'interacted_items'])
    one_hot_encoded_valid, user_dict_one_hot_valid, reversed_user_dict_one_hot_valid  = convert_to_one_hot_encoding_valid_test(valid_interactions_without_target1, reversed_item_dict_one_hot, num_users_train)
    test_target_set = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        #b = data.iloc[index]['baskets'][-1]
        baskets = data.iloc[index]['baskets']
        if baskets:  # Check if the list is not empty
            b = baskets[-1]
        else:
            b = None
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        b1= [b]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b1, data.iloc[index]['last_semester']]
            test_target_set.append(row)
        index +=1
        #if index==30: break
    valid_target_set = pd.DataFrame(test_target_set, columns=['userID', 'baskets', 'last_semester'])
   
    valid_set_all.to_json('./valid_sample_all_v3.json', orient='records', lines=True)
    valid_set_without_target.to_json('./valid_sample_without_target_v3.json', orient='records', lines=True)
    valid_target_set.to_json('./valid_target_set_v3.json', orient='records', lines=True)
    return valid_set_without_target, valid_target_set, valid_set_all, one_hot_encoded_valid, user_dict_one_hot_valid, reversed_user_dict_one_hot_valid

def preprocess_test_data(input_data, item_list, reversed_item_dict_one_hot, num_users_train): #  
  
    data = input_data

    #baskets['num_baskets'] = baskets.baskets.apply(len)
    index=0
    for baskets in data['baskets']:
        new_baskets = []
        ts = []
        tsindex = 0
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_list:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item)
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
                ts.append(data['timestamps'][index][tsindex])
            tsindex += 1
        data['baskets'][index] = new_baskets
        data['num_baskets'][index] = len(new_baskets)
        data['timestamps'][index] = ts
        index +=1  
    users = data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets'], data.iloc[index]['last_semester'], data.iloc[index]['timestamps']]
            test_all.append(row)
        index +=1
        #if index==30: break
    test_set_all = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets', 'last_semester', 'timestamps'])

    test_2 = []
    index = 0
    test_interactions_without_target = []
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']-1, data.iloc[index]['last_semester']]
            test_2.append(row)
            item_ids = []
            for basket in b:
                for item in basket:
                    if item not in item_ids:
                        item_ids.append(item)
            row1 = [user, item_ids]
            test_interactions_without_target.append(row1)
        index +=1
        #if index==30: break
   
    test_set_without_target = pd.DataFrame(test_2, columns=['userID', 'baskets', 'num_baskets', 'last_semester'])
    test_interactions_without_target1 = pd.DataFrame(test_interactions_without_target, columns=['userID', 'interacted_items'])
    one_hot_encoded_test, user_dict_one_hot_test, reversed_user_dict_one_hot_test  = convert_to_one_hot_encoding_valid_test(test_interactions_without_target1, reversed_item_dict_one_hot, num_users_train)
    test_target_set = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        #b = data.iloc[index]['baskets'][-1]
        baskets = data.iloc[index]['baskets']
        if baskets:  # Check if the list is not empty
            b = baskets[-1]
        else:
            b = None
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        b1= [b]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b1, data.iloc[index]['last_semester']]
            test_target_set.append(row)
        index +=1
        #if index==30: break
    test_target_set = pd.DataFrame(test_target_set, columns=['userID', 'baskets', 'last_semester'])
   
    test_set_all.to_json('./test_sample_all_v3.json', orient='records', lines=True)
    test_set_without_target.to_json('./test_sample_without_target_v3.json', orient='records', lines=True)
    test_target_set.to_json('./test_target_set_v3.json', orient='records', lines=True)
    return test_set_without_target, test_target_set, test_set_all, one_hot_encoded_test, user_dict_one_hot_test, reversed_user_dict_one_hot_test


if __name__ == '__main__':
    train_data = pd.read_json('./train_data_all.json', orient='records', lines= True)
    train_set, target_set, total_set, item_list, item_dict, reversed_item_dict,  one_hot_encoded_train, one_hot_encoded_df_train, item_dict_one_hot, reversed_item_dict_one_hot, user_dict_one_hot, one_hot_encoded_train2, user_dict_one_hot_train, reversed_user_dict_one_hot_train = preprocess_data(train_data)
    valid_data = pd.read_json('./valid_data_all.json', orient='records', lines= True)
    dataValid_prev, dataValid_target, dataValid_Total, one_hot_encoded_valid, user_dict_one_hot_valid, reversed_user_dict_one_hot_valid = preprocess_valid_data(valid_data, item_list, reversed_item_dict_one_hot, len(user_dict_one_hot))
    test_data = pd.read_json('./test_data_all.json', orient='records', lines= True)
    dataTest_prev, dataTest_target, datatest_Total, one_hot_encoded_test, user_dict_one_hot_test, reversed_user_dict_one_hot_test = preprocess_test_data(test_data, item_list, reversed_item_dict_one_hot, len(user_dict_one_hot))
    #print(datatest_Total)
    print(item_dict_one_hot)
    print(one_hot_encoded_test)
    print(one_hot_encoded_valid.shape)
