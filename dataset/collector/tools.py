import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
train_size = 0.75 # merge original training set and test set, then split it manually. 
least_samples = 1 # guarantee that each client must have at least one samples for testing. 
alpha = 0.1 # for Dirichlet distribution

def separate_data(data,num_clients,num_classes,niid=False,balance=False,partition=None,class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[i for i in range(num_classes)] for _ in range(num_clients)]
    
    dataset_content,dataset_label = data
    
    if not niid:
        parition = 'pat'
        class_per_client = num_classes
    if partition == 'pat':
        print('not realized')
        # idxs = np.array(range(len(dataset_label)))
        # idx_for_each_class = []
        # for i in range(len(num_classes)):
        #     idx_for_each_class.append(idxs[dataset_label == i])
        
        # class_num_per_client = [class_per_client for _ in range(num_clients)]
        # for i in range(num_clients):
        #     selected_clients = []
        #     for client in range(num_clients):
        #         if class_num_per_client > 0:
        #             selected_clients.append(client)
        #     selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)) * class_per_client)]
            
        #     num_all_samples = len(idx_for_each_class[i])
    if partition == 'dirichlet':
        alphas = [alpha] * num_clients
        #每个客户端获取1个类别的概率为alpha
        label_distribution = np.random.dirichlet(alphas,num_classes)
        y_class_index = [np.argwhere(dataset_label == y).flatten()
                     for y in range(num_classes)]
        #每个客户端对应的样本索引
        clients_dataset_map = [[] for _ in range(num_clients)]
        for class_k,fracs in zip(y_class_index,label_distribution):
            class_size = len(class_k)
            #如此得出类别k在每个一个客户端上能够划分的数量的数组
            splits = (fracs * class_size).astype(int)
            splits[-1] = class_size - splits[:1].sum()
            # class_k = np.random.shuffle(class_k)
            #np.split是根据边界来进行划分，比如三个客户端分别划分10,20,15个数据集，
            # 那么np.split输入的数组应该为[10,30,45]
            cumulative_sum = np.cumsum(splits)
            #idcs是个二维数组
            idcs = np.split(class_k,cumulative_sum)
            idcs = idcs[:-1]
            for i,idx in enumerate(idcs):
                clients_dataset_map[i].append(idx)
    # for (i,client_data) in enumerate(clients_dataset_map):
    #     for idx in client_data:
    #         for j in idx:
    #             X[i].append(dataset_content[j])
    #             y[i].append(dataset_label[j])
    #             for v in j:
    #                 class_type = int(dataset_label[v])
    #                 statistic[i][class_type] += 1  
    
    for (i,client_i_data_idx) in enumerate(clients_dataset_map):
        for (j,class_k_index) in enumerate(client_i_data_idx):
            for index in class_k_index:
                X[i].append(dataset_content[index])
                y[i].append(dataset_label[index])
            # statistic is counting in client i how many dataset in class j?
            statistic[i][j] += len(class_k_index)   
            
            
    del data
    
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of each labels: ", [i for i in statistic[client]])
        print("-" * 50)
    return X,y,statistic
        

def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, 
                num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients, 
        'num_classes': num_classes, 
        'non_iid': niid, 
        'balance': balance, 
        'partition': partition, 
        'Size of samples for labels in clients': statistic, 
        'alpha': alpha, 
        'batch_size': batch_size, 
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")

def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[]}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data