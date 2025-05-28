import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import random

#def split_data(input_file):
def preprocess_train_data_part1(input_data):
    data = input_data
    data["userID"] = data["userID"].fillna(-1).astype('int32')
    data=data[data["userID"]!=-1].reset_index(drop=True)

    itemIDs = {}
    index=0
    for baskets in data['baskets']:
        new_baskets = []
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item not in itemIDs:
                    itemIDs[item] = len(itemIDs)+1  # indexing starts from 0
                new_basket.append(itemIDs[item])
            new_baskets.append(new_basket)
        data['baskets'][index] = new_baskets
        index +=1                

    # itemID_values = data["itemID"].astype('str')
    # sorting by itemIDs
    # itemIDs = sorted(set(itemID_values))
    # print("itemIDs: ", itemIDs)
    # converting itemIDs into index from 0 to length(item_list)-1
    # itemIDs = {c: i for (i, c) in enumerate(itemIDs)}
    # print("itemIDs: ", itemIDs)
    
    reversed_item_dict = dict(zip(itemIDs.values(), itemIDs.keys()))
    # print("reversed_item_dict: ", reversed_item_dict)
    # data["itemID"] = itemID_values.map(itemIDs).astype('int32')

    # users = sorted(set(data["userID"]))

    # #converting userIDs into index from 0 to length(user_list)-1
    # users_dict = {c: i for (i, c) in enumerate(users)}
    # reversed_user_dict = dict(zip(users_dict.values(), users_dict.keys()))
    # #print("reversed_user_dict: ", reversed_user_dict)
    # data["userID"] = data["userID"].map(users_dict)
    user_dict = {}
    index = 0
    len1 =0
    reversed_user_dict = {}
    for user in data["userID"]:
        user_dict[user] = len1
        data['userID'][index] = user_dict[user]
        index += 1
        len1+=1
        reversed_user_dict[user_dict[user]]= user
    return data, itemIDs, user_dict, reversed_item_dict, reversed_user_dict
    
def preprocess_train_data_part2(input_data):

    # baskets['num_baskets'] = baskets.baskets.apply(len)
    data = input_data
    users = data.userID.values
    max_len = 0
    max_basket_size = 0
    train_valid = []
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        max_len = max(max_len, data.iloc[index]['num_baskets'])
        for bskt in b:
            max_basket_size = max(max_basket_size, len(bskt))
        row = [user, b, data.iloc[index]['num_baskets']]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        train_valid.append(row)
        
    train_set_all = pd.DataFrame(train_valid, columns=['userID', 'baskets', 'num_baskets'])

    train_valid2 = []
    repeat_ratio_u = 0
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        #if baskets.iloc[index]['num_baskets']>=2:
        row = [user, b, data.iloc[index]['num_baskets']-1]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        train_valid2.append(row)
        item_ids = []
        for basket in b:
            for item in basket:
                if item not in item_ids:
                    item_ids.append(item)
        b2 = data.iloc[index]['baskets'][-1]
        cnt5 = 0
        for item2 in b2:
            if item2 in item_ids:
                cnt5 += 1
        repeat_ratio_u += (cnt5/len(b2))
    avg_repeat_ratio = repeat_ratio_u/ len(data) 
        
    train_set_without_target = pd.DataFrame(train_valid2, columns=['userID', 'baskets', 'num_baskets'])

    target_set = []
    for user in users:
        index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][-1]
        #if baskets.iloc[index]['num_baskets']>=2:
        row = [user, b]
        #row = [user, b, baskets.iloc[index]['num_baskets']]
        target_set.append(row)
    
    target_set = pd.DataFrame(target_set, columns=['userID', 'baskets'])
    pickle_file = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_set_all.pickle'
    with open(pickle_file, 'wb') as handle:
        train_samples = {}
        #total_items = set()
        users = train_set_all.userID.values
        #print(users)
        #time_baskets = train_set_all.baskets.values

        for u in tqdm(users):
            u_baskets = train_set_all[train_set_all['userID'] == u].baskets.values[0]  
            train_samples[u] = u_baskets
        pickle.dump(train_samples, handle)
   
    train_set_all.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_sample_all_cr_v2.json', orient='records', lines=True)
    train_set_without_target.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_set_without_target_cr_v2.json', orient='records', lines=True)
    target_set.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/target_set_cr_v2.json', orient='records', lines=True)
    return train_set_all, train_set_without_target, target_set, max_len, avg_repeat_ratio, max_basket_size
    #return train_set_all, train_set_without_target, target_set, max_len

def preprocess_valid_data_part1(input_data, reversed_user_dict, item_dict): #
    data = input_data
    # data["userID"] = data["userID"].fillna(-1).astype('int32')
    # data=data[data["userID"]!=-1].reset_index(drop=True)
    #itemIDs = {}
    #user_dict ={}
    #item_dict = {}
    index=0
    for baskets in data['baskets']:
        new_baskets = []
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_dict:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item_dict[item])
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
        data['baskets'][index] = new_baskets
        data['num_baskets'][index] = len(new_baskets)
        index +=1    

    
    #reversed_user_dict = dict(zip(user_dict.values(), user_dict.keys()))
    len1 = len(reversed_user_dict)
    user_dict2 = {}
    index = 0
    reversed_user_dict2 = {}
    for user in data["userID"]:
        user_dict2[user] = len1
        data['userID'][index] = user_dict2[user]
        index += 1
        len1+=1
        reversed_user_dict2[user_dict2[user]]= user
    # reversed_user_dict2 = dict(zip(user_dict2.values(), user_dict2.keys()))


    #reversed_user_dict2 = dict(zip(user_dict2.values(), user_dict2.keys()))
    #print("reversed_user_dict: ", reversed_user_dict)
    #data["userID"] = data["userID"].map(users)

    return data, user_dict2, reversed_user_dict2

def preprocess_valid_data_part2(input_data):
    data = input_data
    users = data.userID.values
    valid_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']]
            #row = [user, b, baskets.iloc[index]['num_baskets']]
            valid_all.append(row)
        index += 1
    valid_set_all = pd.DataFrame(valid_all, columns=['userID', 'baskets', 'num_baskets'])
    valid_2 = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']-1]
            valid_2.append(row)
        index += 1
   
    valid_set_without_target = pd.DataFrame(valid_2, columns=['userID', 'baskets', 'num_baskets'])
    validation_target_set = []
    index = 0
    for user in data.userID.values:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][-1]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b]
            validation_target_set.append(row)
        index += 1
    validation_target_set = pd.DataFrame(validation_target_set, columns=['userID', 'baskets'])

    pickle_file = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/valid_set_all.pickle'
    with open(pickle_file, 'wb') as handle:
        valid_samples = {}
        #total_items = set()
        users = valid_set_all.userID.values
        #print(users)
        #time_baskets = valid_set_all.baskets.values

        for u in tqdm(users):
            u_baskets = valid_set_all[valid_set_all['userID'] == u].baskets.values[0]  
            valid_samples[u] = u_baskets
        pickle.dump(valid_samples, handle)
    
    valid_set_all.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/valid_sample_all_cr_v2.json', orient='records', lines=True)
    valid_set_without_target.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/valid_sample_without_target_cr_v2.json', orient='records', lines=True)
    validation_target_set.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/validation_target_set_cr_v2.json', orient='records', lines=True)
    return valid_set_all, valid_set_without_target, validation_target_set
    #return valid_set_all, valid_set_without_target, validation_target_set

def preprocess_test_data_part1(input_data, reversed_user_dict, item_dict, reversed_user_dict2): #  
  
    data = input_data
    #user_dict ={}
    #item_dict = {}
    # data["userID"] = data["userID"].fillna(-1).astype('int32')
    # data=data[data["userID"]!=-1].reset_index(drop=True)
    #itemIDs = {}
    index=0
    for baskets in data['baskets']:
        new_baskets = []
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item in item_dict:
                    #item_dict[item] = len(item_dict)
                    new_basket.append(item_dict[item])
            if(len(new_basket)>0):
                new_baskets.append(new_basket)
        data['baskets'][index] = new_baskets
        data['num_baskets'][index] = len(new_baskets)
        index +=1    
    
    #reversed_user_dict = dict(zip(user_dict.values(), user_dict.keys()))
    len1 = len(reversed_user_dict) + len(reversed_user_dict2)
    user_dict3 = {}
    index = 0
    reversed_user_dict3 = {}
    for user in data["userID"]:
        user_dict3[user] = len1
        data['userID'][index] = user_dict3[user]
        index += 1
        len1+=1
        reversed_user_dict3[user_dict3[user]]= user

    #reversed_user_dict3 = dict(zip(user_dict3.values(), user_dict3.keys()))
    # baskets = data.groupby(['userID', 'timestamp'])['itemID'].apply(list).reset_index()
    # baskets = baskets.groupby(['userID'])['itemID'].apply(list).reset_index()  #
    # baskets.columns = ['userID', 'baskets']

    #baskets['num_baskets'] = baskets.baskets.apply(len)

    return data, user_dict3, reversed_user_dict3

def preprocess_test_data_part2(input_data):
    data = input_data
    users = data.userID.values
    test_all = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:]
        #b = baskets.iloc[index]['baskets'][0:]
        #if baskets.iloc[index]['num_baskets']>=2:
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']]
            test_all.append(row)
        index +=1
    test_set_all = pd.DataFrame(test_all, columns=['userID', 'baskets', 'num_baskets'])

    test_2 = []
    index = 0
    for user in users:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][0:-1]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b, data.iloc[index]['num_baskets']-1]
            test_2.append(row)
        index +=1
   
    test_set_without_target = pd.DataFrame(test_2, columns=['userID', 'baskets', 'num_baskets'])
    test_target_set = []
    index = 0
    for user in data.userID.values:
        #index = data[data['userID'] == user].index.values[0]
        b = data.iloc[index]['baskets'][-1]
        #b = baskets.iloc[index]['baskets'][0:]
        if data.iloc[index]['num_baskets']>=3:
            row = [user, b]
            test_target_set.append(row)
        index +=1
    test_target_set = pd.DataFrame(test_target_set, columns=['userID', 'baskets'])
    pickle_file = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_set_all.pickle'
    with open(pickle_file, 'wb') as handle:
        test_samples = {}
        #total_items = set()
        users = test_set_all.userID.values
        #print(users)
        #time_baskets = test_set_all.baskets.values

        for u in tqdm(users):
            u_baskets = test_set_all[test_set_all['userID'] == u].baskets.values[0] 
            test_samples[u] = u_baskets
        pickle.dump(test_samples, handle)
   
    test_set_all.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_sample_all_cr_v2.json', orient='records', lines=True)
    test_set_without_target.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_sample_without_target_cr_v2.json', orient='records', lines=True)
    test_target_set.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_target_set_cr_v2.json', orient='records', lines=True)
    # test_set_all.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_sample_all_with_more_target_baskets.json', orient='records', lines=True)
    # test_set_without_target.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_sample_without_target_with_more_target_baskets.json', orient='records', lines=True)
    # test_target_set.to_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_target_set_with_more_target_baskets.json', orient='records', lines=True)
    return test_set_all, test_set_without_target, test_target_set
    #return test_set_all, test_set_without_target, test_target_set
def negative_sample(pickle_file, train_data_all, valid_sample_without_target, test_sample_without_target):
    """
    Create negative sample.

    Args:
        pickle_file:
    Returns:
         (key: values) -> (userID: negative samples for the user)
    """
    #pickle_file = '/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/neg_sample_cr_v2.pickle'
    with open(pickle_file, 'wb') as handle:
        # train_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_sample_all.json', orient='records', lines= True)
        # #train_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN_2/train_sample_main.json', orient='records', lines= True)
        # #valid_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN_2/validation_sample_main.json', orient='records', lines=True)
        # valid_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/valid_sample_without_target.json', orient='records', lines=True)
        # test_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_sample_without_target.json', orient='records', lines=True)
        train_data = train_data_all
        valid_data = valid_sample_without_target
        test_data = test_sample_without_target

        #total_data = train_data.append(valid_data)
        total_data = pd.concat([train_data, valid_data], ignore_index=True)
        total_data = pd.concat([total_data, test_data], ignore_index=True)
        #total_data = pd.concat(train_data, valid_data)
        neg_samples = {}
        total_items = set()
        users = total_data.userID.values
        #print(users)
        time_baskets = total_data.baskets.values

        for baskets in tqdm(time_baskets):
            for basket in baskets:
                for item in basket:
                    total_items.add(item)

        for u in tqdm(users):
            history_items = set()
            u_baskets = total_data[total_data['userID'] == u].baskets.values[0]  
            for basket in u_baskets:
                for item in basket:
                    history_items.add(item)
            neg_items = total_items - history_items
            neg_samples[u] = neg_items
            #print(u, " ",neg_samples[u])
        pickle.dump(neg_samples, handle)


if __name__ == '__main__':
   #train, test, valid = split_data('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/CDREAM_LGCN_2/train_sample.csv')
   #train_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Filtered_data/train_sample_augmented_CR.json', orient='records', lines= True)
   #train_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/train_data_all_CR.json', orient='records', lines= True)
   train_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/train_data_all_CR.json', orient='records', lines= True)
   # train_data_all_CR
   train_data, item_dict, user_dict, reversed_item_dict, reversed_user_dict = preprocess_train_data_part1(train_data)
   train_all, train_set_without_target, target, max_len, repeat_ratio, max_basket_size = preprocess_train_data_part2(train_data) 
   #print(len(item_dict))
#    print(train_all)
#    print("max_len:", max_len)
   #print(target)
   #valid_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/valid_data_all.json', orient='records', lines= True)
   #valid_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/valid_data_all_CR.json', orient='records', lines= True)
   valid_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/valid_data_all_CR.json', orient='records', lines= True)
   valid_data, user_dict2, reversed_user_dict2 = preprocess_valid_data_part1(valid_data, reversed_user_dict, item_dict)
   valid_all, valid_set_without_target, valid_target = preprocess_valid_data_part2(valid_data) #  #, 
   #print("reversed_user_dict2: ", reversed_user_dict2)
   #print(valid_all)
#    test_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/test_data_all_CR.json', orient='records', lines= True)
   #test_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/test_data_all_CR.json', orient='records', lines= True)
   test_data = pd.read_json('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/test_data_all_CR.json', orient='records', lines= True)
   test_data, user_dict3, reversed_user_dict3 = preprocess_test_data_part1(test_data, reversed_user_dict, item_dict, reversed_user_dict2)
   test_all, test_set_without_target, test_target = preprocess_test_data_part2(test_data) #, item_dict, user_dict, reversed_item_dict, reversed_user_dict #, 
   #print(test_all) 
   negative_sample('/a/bear.cs.fiu.edu./disk/bear-b/users/mkhan149/Downloads/Experiments/Others/MMNR/neg_sample_cr_v2.pickle', train_all, valid_set_without_target, test_set_without_target)
   print("max_len: ", max_len)
   print("max_basket_size: ", max_basket_size)